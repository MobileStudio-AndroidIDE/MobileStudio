"""Context and reference resolution: what a referring expression points AT.

The turn's referring expressions ("that", "it", "they", "the last
topic") are resolved against the LOCAL semantic context first — the
clause, the sentence, the paragraph and the rest of the request — and
only then against the conversation's state.  They are never sent to a
retriever as query text, and a pronoun whose antecedent stands in the
request itself never becomes a clarification:

    current clause → current sentence → current paragraph →
    current request → conversation context
        ↓  Recency / Type Compatibility / Semantic Compatibility /
           Discourse Role
    Reference { surface_form, refers_to, target, confidence, evidence }
        ↓  confidence below the clarification threshold
    clarification (the referent is ASKED for, never guessed)

The scorer is general: it reads the utterance's act and the candidate's
kind — there is no per-sentence branch, no keyword→referent table, and
no special case for a particular surface form.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# Below this confidence an anaphoric reference is not committed: the
# turn asks the speaker which referent they meant instead of guessing.
CLARIFICATION_THRESHOLD = 0.45

# Two candidates this close together are ambiguous: a winner by a
# hair is not a resolution.
AMBIGUITY_MARGIN = 0.05

# The scoring weights — one per resolution signal.  They are properties
# of reference resolution itself, not of any utterance.
_WEIGHTS = {
    "recency": 0.35,
    "type_compatibility": 0.25,
    "semantic_compatibility": 0.25,
    "discourse_role": 0.15,
}


@dataclass
class CandidateReference:
    """One antecedent the resolution considered."""
    target: str = ""
    kind: str = "topic"        # topic | proposition | entity | utterance
    source: str = ""           # topic_state | previous_turn | utterance
    recency: float = 0.0
    type_compatibility: float = 0.0
    semantic_compatibility: float = 0.0
    discourse_role: float = 0.0
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind,
            "source": self.source,
            "score": round(self.score, 3),
            "signals": {
                "recency": round(self.recency, 3),
                "type_compatibility": round(self.type_compatibility, 3),
                "semantic_compatibility": round(self.semantic_compatibility, 3),
                "discourse_role": round(self.discourse_role, 3),
            },
            "reasons": list(self.reasons),
        }


@dataclass
class Reference:
    """The resolved referring expression (or the honest failure to)."""
    surface_form: str = ""        # the expression itself ("that")
    refers_to: str = ""           # utterance | previous_topic |
                                  # previous_turn | unresolved
    target: str = ""              # the referent the target names
    target_kind: str = ""
    contrast_with: str = ""       # the pair reference's OTHER term
                                  # ("the other one" → its counterpart)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    candidates: list[CandidateReference] = field(default_factory=list)
    needs_clarification: bool = False

    @property
    def resolved(self) -> bool:
        return bool(self.target) and not self.needs_clarification

    def to_dict(self) -> dict:
        return {
            "surface_form": self.surface_form,
            "refers_to": self.refers_to,
            "target": self.target,
            "target_kind": self.target_kind,
            "contrast_with": self.contrast_with,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
            "needs_clarification": self.needs_clarification,
            "candidates": [c.to_dict() for c in self.candidates],
        }


@dataclass
class TopicState:
    """The discourse's topical state — what the conversation is about.

    A new clear topic becomes `current_topic` (the previous one is
    retained in `previous_topics`, never destroyed: a reference may
    still point back at it); a reset act retires the current topic.
    """
    current_topic: str = ""
    previous_topics: list[str] = field(default_factory=list)
    active_entities: list[str] = field(default_factory=list)
    active_concepts: list[str] = field(default_factory=list)
    active_task: str = ""
    language: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)

    def push(self, topic: str, task: str = "", entities=(),
             concepts=(), language: str = "", confidence: float = 0.0,
             evidence=()) -> None:
        """The turn's referent becomes the current topic."""
        topic = (topic or "").strip()
        if topic and topic != self.current_topic:
            if self.current_topic:
                self.previous_topics.append(self.current_topic)
                del self.previous_topics[:-8]
        if topic:
            self.current_topic = topic
        if task:
            self.active_task = task
        if entities:
            self.active_entities = _dedupe(
                list(entities) + self.active_entities)[:12]
        if concepts:
            self.active_concepts = _dedupe(
                list(concepts) + self.active_concepts)[:12]
        if language:
            self.language = language
        if confidence:
            self.confidence = float(confidence)
        if evidence:
            self.evidence = list(evidence)[-6:]

    def retire(self, evidence=()) -> None:
        """A reset_topic discourse act: the current topic is retired."""
        if self.current_topic:
            self.previous_topics.append(self.current_topic)
            del self.previous_topics[:-8]
        self.current_topic = ""
        self.active_task = ""
        if evidence:
            self.evidence = list(evidence)[-6:]

    @property
    def candidates(self) -> list[str]:
        """The topics a reference may resolve against, newest first."""
        out = [t for t in [self.current_topic] + list(reversed(self.previous_topics)) if t]
        return out

    def to_dict(self) -> dict:
        return {
            "current_topic": self.current_topic,
            "previous_topics": list(self.previous_topics),
            "active_entities": list(self.active_entities),
            "active_concepts": list(self.active_concepts),
            "active_task": self.active_task,
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "evidence": list(self.evidence),
        }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


# The referent kind an act SELECTS for: what the speaker is doing
# constrains what the reference can mean (type compatibility).
_ACT_EXPECTED_KIND = {
    "explanation_request": {"topic", "proposition"},
    "simplify_explanation": {"topic", "proposition"},
    "information_query": {"topic", "proposition"},
    "analysis": {"proposition"},
    "command_query": {"topic"},
    "recommendation_request": {"topic", "proposition"},
    "code_generation": {"entity"},
    "code_modification": {"entity"},
    "project_generation": {"entity"},
    "clarification": {"topic", "proposition", "entity"},
}

# The kinds an act may still accept when it declares none (honest
# fallback: an unexpected kind lowers the score, it never deletes it).
_DEFAULT_KINDS = {"topic", "proposition", "entity", "utterance"}

# The social/conversational acts: they answer from the conversation's
# own structure, so an unresolved referent never turns them into a
# clarification (a "thanks, that helps" is acknowledged without its
# antecedent being recovered).
_SOCIAL_ACTS = frozenset({
    "greeting", "conversation", "clarification", "identity",
    "capability_query", "language_capability_query",
})

# The acts answered from KNOWLEDGE about a referent: they need
# something to be about.  A turn of one of these that names no referent
# and resolves none asks for it (a clarification) — it never answers
# about whatever the retrieval happens to match.
_CONTENT_ACTS = frozenset({
    "information_query", "explanation_request", "simplify_explanation",
    "analysis", "analysis_request", "recommendation_request",
    "suitability_query", "command_query",
})


class ReferenceResolver:
    """Resolve a turn's referring expressions against the discourse.

    General over surface forms: the utterance's anaphoric class (from
    the act layer's grammar) tells us a reference EXISTS; the discourse
    state supplies the candidates; the scoring decides which one it is.
    """

    def __init__(self, knowledge_lookup=None):
        # knowledge_lookup(query) -> [(concept, score)] : whether a
        # candidate is a knowledge-worthy topic (semantic compatibility).
        self.knowledge_lookup = knowledge_lookup

    # -- candidate construction ------------------------------------

    def _candidates_from(self, semantic, topic_state, history,
                         utterance_topics: list[str]) -> list[CandidateReference]:
        out: list[CandidateReference] = []
        state = topic_state or TopicState()
        for index, topic in enumerate(state.candidates):
            recency = 1.0 if index == 0 else max(0.3, 0.75 ** index)
            out.append(CandidateReference(
                target=topic, kind="topic", source="topic_state",
                recency=recency))
        seen = {c.target.strip().lower() for c in out}

        def _add(target: str, kind: str, recency: float) -> None:
            text = str(target or "").strip()
            if not text or text.lower() in seen:
                return
            seen.add(text.lower())
            out.append(CandidateReference(
                target=text, kind=kind, source="previous_turn",
                recency=recency))

        turns = list(history or [])[-3:]
        for offset, turn in enumerate(reversed(turns)):
            text = _turn_text(turn)
            recency = max(0.2, 0.6 - 0.15 * offset)
            structured = _turn_state(turn)
            if structured:
                # The turn's STRUCTURED meaning — what it was about, what
                # it carried and what the response SAID.  The antecedent
                # pool is the record, never a re-parse of the raw text.
                for topic in _state_values(structured, "topic", "topics"):
                    _add(topic, "topic", recency)
                semantic_record = structured.get(
                    "semantic_representation") or {}
                if isinstance(semantic_record, dict):
                    for topic in _state_values(
                            {"topics": semantic_record.get("topics")},
                            "topics"):
                        _add(topic, "topic", recency)
                for entity in _state_values(structured, "entities"):
                    _add(entity, "entity", recency)
                # The activated knowledge concepts stay in the state but
                # never become referents: they are the knowledge
                # layer's own activations (ids such as "lang_python"),
                # not things the conversation named.
                kind, referent = _state_referent(structured)
                if referent:
                    _add(referent, kind or "topic", recency)
                claims = _state_claims(structured)
                for claim in claims:
                    _add(claim, "proposition", max(0.15, recency - 0.1))
                if not claims and text:
                    _add(_proposition_of(text), "proposition",
                         max(0.15, recency - 0.1))
                continue
            # A turn recorded before the structured record existed: its
            # topics and its own text are all the discourse kept.
            for salient in _turn_topics(turn):
                _add(salient, "topic", recency)
            if text:
                # the previous utterance itself is a fallback referent
                _add(_proposition_of(text), "proposition",
                     max(0.15, recency - 0.1))
        for topic in utterance_topics:
            if topic and topic.strip().lower() not in seen:
                out.append(CandidateReference(
                    target=topic, kind="utterance", source="utterance",
                    recency=1.0))
        return [c for c in out if c.target]

    # -- scoring ----------------------------------------------------

    def _score(self, candidate: CandidateReference, intent_type: str,
               focused_topic: str) -> None:
        expected = _ACT_EXPECTED_KIND.get(intent_type) or _DEFAULT_KINDS
        if candidate.kind in expected:
            candidate.type_compatibility = 1.0
            candidate.reasons.append(
                f"kind={candidate.kind} fits {intent_type or 'the act'}")
        else:
            candidate.type_compatibility = 0.4
            candidate.reasons.append(
                f"kind={candidate.kind} is not the act's referent kind")

        semantic_score = 0.0
        if focused_topic and _same_referent(candidate.target, focused_topic):
            semantic_score = 1.0
            candidate.reasons.append("matches the turn's topic focus")
        elif self.knowledge_lookup is not None:
            try:
                hits = self.knowledge_lookup(candidate.target) or []
            except Exception:
                hits = []
            if hits:
                semantic_score = 0.8
                candidate.reasons.append(
                    "candidate is a known knowledge topic")
        candidate.semantic_compatibility = semantic_score

        # discourse role: the current topic IS the discourse focus; a
        # candidate from the previous turn is one step further back.
        if candidate.source == "topic_state" and candidate.recency >= 0.99:
            candidate.discourse_role = 1.0
            candidate.reasons.append("is the current discourse topic")
        elif candidate.source == "previous_turn":
            candidate.discourse_role = 0.6
            candidate.reasons.append("was salient in the previous turn")
        else:
            candidate.discourse_role = 0.4

        candidate.score = (
            _WEIGHTS["recency"] * candidate.recency
            + _WEIGHTS["type_compatibility"] * candidate.type_compatibility
            + _WEIGHTS["semantic_compatibility"] * candidate.semantic_compatibility
            + _WEIGHTS["discourse_role"] * candidate.discourse_role
        )

    def _resolve_pair(self, pair: str, candidates, focused, topic_state,
                      social):
        """Resolve a PAIR reference ("the other one") to the discourse's
        other topic, carrying its counterpart so a contrast can be
        composed — or honestly fail when the discourse holds fewer than
        two topics to pair."""
        if social:
            return None
        topics = []
        for c in candidates:
            if c.kind != "topic":
                continue
            if not any(_same_referent(c.target, t.target) for t in topics):
                topics.append(c)
        if len(topics) < 2:
            return Reference(
                surface_form=pair, refers_to="unresolved", target="",
                confidence=0.4, candidates=candidates,
                needs_clarification=True,
                evidence=[f"'{pair}' pairs two discourse topics; the "
                          f"discourse holds {len(topics)}"])
        focus = str(focused or "").strip()
        if not focus:
            focus = str(getattr(topic_state, "current_topic", "")
                        or "").strip()
        others = [t for t in topics
                  if focus and not _same_referent(t.target, focus)]
        if not others:
            others = topics[1:]
        best = others[0]
        counterpart = (focus
                       if focus and not _same_referent(best.target, focus)
                       else next((t.target for t in topics
                                  if not _same_referent(t.target,
                                                        best.target)), ""))
        return Reference(
            surface_form=pair, refers_to=_refers_to(best.source),
            target=best.target, target_kind=best.kind,
            contrast_with=counterpart,
            confidence=min(0.9, max(float(best.score or 0.0), 0.6)),
            candidates=candidates,
            evidence=[f"'{pair}' → {best.target} (the pair's other "
                      f"member; counterpart {counterpart or 'none'})"])

    # -- the resolution ---------------------------------------------

    def resolve(self, semantic, request: str, topic_state=None,
                history=(), surface_forms=()) -> Reference:
        """The turn's reference: resolved, or honestly unresolved."""
        intent_type = str(((getattr(semantic, "dialogue_intent", None)
                            or {}).get("type")) or "")
        social = intent_type in _SOCIAL_ACTS
        utterance_topics = [str(t) for t in
                            (getattr(semantic, "intent_topics", None) or [])
                            if t]
        focused = utterance_topics[0] if utterance_topics else ""

        if surface_forms:
            anaphora = list(surface_forms)
        else:
            anaphora = _anaphora_in(_discourse_text(request))
        if not anaphora:
            # No referring expression: the turn either names its
            # referent itself, or names none — and then the topic
            # layer's own record says WHERE the referent came from
            # (the discourse's topic state, for an elliptical request).
            if focused:
                source = _reference_source(semantic)
                if source == "discourse_topic":
                    # The discourse supplied the referent: the topic
                    # state's own candidates are what it was read
                    # against, so the resolution is observable.
                    candidates = self._candidates_from(
                        semantic, topic_state, history, [])
                    for candidate in candidates:
                        self._score(candidate, intent_type, focused)
                    matched = [c for c in candidates
                               if _same_referent(c.target, focused)]
                    matched.sort(key=lambda c: -c.score)
                    if matched:
                        best = matched[0]
                        return Reference(
                            surface_form="",
                            refers_to=_refers_to(best.source),
                            target=focused, target_kind=best.kind,
                            confidence=min(0.95, best.score),
                            evidence=[
                                "the turn names no referent: the "
                                f"discourse's topic state supplies {focused}"],
                            candidates=candidates,
                        )
                return Reference(
                    surface_form="", refers_to="utterance", target=focused,
                    target_kind="utterance",
                    confidence=min(0.95, 0.7 + 0.25 * float(
                        getattr(semantic, "confidence", 0.0) or 0.0)),
                    evidence=["no anaphoric expression: the utterance "
                              "supplies its own referent"],
                    candidates=[CandidateReference(
                        target=focused, kind="utterance",
                        source="utterance", recency=1.0,
                        type_compatibility=1.0, semantic_compatibility=1.0,
                        discourse_role=1.0, score=1.0,
                        reasons=["the turn's own content"])],
                )
            # No referent anywhere: only a content act NEEDS one — a
            # greeting or a capability question never becomes a
            # clarification for lacking a topic.
            return Reference(
                surface_form="", refers_to="unresolved",
                needs_clarification=(
                    not social and intent_type in _CONTENT_ACTS
                    and bool(request.strip())),
                evidence=["no referent in the utterance and "
                          "none resolved from the discourse"])

        # The REQUEST's own referent first: a pronoun with a local
        # antecedent is resolved HERE — clause, then sentence, then the
        # rest of the request — and no conversation candidate may
        # outrank it (goal: current clause > sentence > paragraph >
        # request > conversation).
        if not _pair_reference(anaphora):
            local = _local_antecedent(request, anaphora[0])
            if local is not None:
                nominal, where, score = local
                return Reference(
                    surface_form=anaphora[0], refers_to="utterance",
                    target=nominal, target_kind="clause",
                    confidence=score,
                    evidence=[
                        f"'{anaphora[0]}' → {nominal} "
                        f"(local antecedent: {where} of this request)"],
                    candidates=[CandidateReference(
                        target=nominal, kind="clause",
                        source="utterance", recency=1.0,
                        type_compatibility=1.0,
                        semantic_compatibility=1.0,
                        discourse_role=1.0, score=score,
                        reasons=[f"local antecedent in the {where}"])],
                    needs_clarification=False)

        candidates = self._candidates_from(
            semantic, topic_state, history,
            # a same-utterance antecedent precedes the anaphor: it is a
            # candidate only when it precedes the reference.
            [t for t in utterance_topics
             if not _follows_the_reference(t, request, anaphora)])
        if not candidates:
            return Reference(
                surface_form=anaphora[0], refers_to="unresolved",
                target=focused, target_kind="utterance" if focused else "",
                confidence=0.5 if focused else 0.0,
                needs_clarification=bool(request.strip()) and not social,
                evidence=[f"'{anaphora[0]}' has no antecedent in the "
                          "discourse"])

        for candidate in candidates:
            self._score(candidate, intent_type, focused)
        candidates.sort(key=lambda c: -c.score)

        # A PAIR reference names a role, not a referent: it takes up
        # the OTHER member of two the discourse holds.  The roles are
        # filled from the DISCOURSE's topics; with fewer than two the
        # expression cannot be resolved and the turn asks back.
        pair = _pair_reference(anaphora)
        if pair:
            paired = self._resolve_pair(pair, candidates, focused,
                                        topic_state, social)
            if paired is not None:
                return paired

        best = candidates[0]
        confidence = best.score
        ambiguous = (len(candidates) > 1
                     and candidates[1].score >= best.score - AMBIGUITY_MARGIN)
        if ambiguous:
            confidence = min(confidence, 0.4)

        if confidence < CLARIFICATION_THRESHOLD:
            return Reference(
                surface_form=anaphora[0], refers_to="unresolved",
                target=focused if social else "",
                target_kind="utterance" if social and focused else "",
                confidence=confidence,
                evidence=[f"'{anaphora[0]}' could not be resolved above "
                          f"the {CLARIFICATION_THRESHOLD} threshold"],
                candidates=candidates,
                needs_clarification=not social)

        if best.source == "topic_state" and best.recency >= 0.99:
            refers_to = "previous_topic"
        elif best.source == "utterance":
            refers_to = "utterance"
        else:
            refers_to = "previous_turn"
        return Reference(
            surface_form=anaphora[0], refers_to=refers_to,
            target=best.target, target_kind=best.kind,
            confidence=confidence,
            evidence=[f"'{anaphora[0]}' → {best.target} "
                      f"({'; '.join(best.reasons)})"],
            candidates=candidates,
            needs_clarification=False)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

# The anaphoric class (grammar): demonstratives, personal pronouns and
# the discourse-recipient forms.  Membership is a grammatical class, not
# a keyword→answer table: nothing here says WHAT a form refers to.
_ANAPHOR_WORDS = (
    "that", "this", "it", "they", "them", "those", "these",
    "the former", "the latter", "그거", "그것", "이거", "이것", "저거",
    "아까 그", "말한 것", "지난 주제",
)

# The multi-word recipient references ("the previous topic", "what we
# discussed", "the result", "that function", "that code").
_ANAPHOR_PHRASES = (
    "the previous topic", "the last topic", "the topic", "the last thing",
    "what you mentioned", "what we discussed", "what i mentioned",
    "the above", "the result", "that function", "that code",
    "your earlier", "the earlier",
    "이전 주제", "지난 주제", "아까 말한", "방금 그", "방금 전",
)


# The PAIR reference (grammar): an expression that takes up the OTHER
# member of a two-member set the discourse holds.  It does not name a
# referent — it names a ROLE in a pair, so the resolution needs two
# discourse referents to fill the roles; with fewer it asks back.
_PAIR_REFERENCES = (
    "the other one", "the other", "the former", "the latter",
    "the other of them", "the other two",
    "다른 하나", "그 다른 것", "다른 거", "나머지 하나", "그 반대",
    "이전 것", "앞의 것",
)


# ---------------------------------------------------------------------
# the LOCAL antecedent: what the request ITSELF points at
# ---------------------------------------------------------------------

# The closed class: grammar words never stand as a nominal of their own.
_LOCAL_FUNCTION = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "your",
    "his", "her", "its", "our", "their", "some", "any", "each", "every",
    "all", "both", "either", "neither", "one", "two", "many", "few",
    "i", "me", "we", "us", "you", "he", "him", "she", "them", "who",
    "whom", "whose", "which", "what", "where", "when", "why", "how",
    "and", "but", "or", "nor", "so", "yet", "if", "then", "than",
    "because", "although", "though", "unless", "while", "when", "where",
    "as", "of", "to", "in", "on", "at", "by", "for", "from", "with",
    "about", "into", "over", "under", "between", "through", "during",
    "before", "after", "above", "below", "up", "down", "out", "off",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "shall", "should", "may", "might", "must",
    "그", "이", "저", "그리고", "하지만", "또는", "은", "는", "이", "가",
    "을", "를", "에", "의", "로", "에서", "까지", "보다", "하고",
})

# The predicate lexicon the act layer classifies with: a word in it
# predicates the clause — it is never the clause's subject.
_PREDICATE_WORDS: frozenset | None = None


def _predicate_words() -> frozenset:
    global _PREDICATE_WORDS
    if _PREDICATE_WORDS is None:
        from brain.dialogue_act import _PREDICATE_CLASSES
        words = set()
        for group in _PREDICATE_CLASSES.values():
            for entry in group:
                for token in re.findall(r"[a-z]{3,}", str(entry).lower()):
                    words.add(token)
        words.update({
            "happen", "happens", "occurred", "occur", "die", "dies",
            "change", "changes", "grow", "grows", "born", "begin",
            "begins", "began", "end", "ends", "last", "lasts", "rise",
            "rises", "fall", "falls", "form", "forms", "formed",
            "create", "creates", "created", "work", "works", "worked",
            "learn", "learns", "study", "studies", "teach", "teaches",
        })
        _PREDICATE_WORDS = frozenset(words)
    return _PREDICATE_WORDS


# The clause boundary: a coordinate or subordinate clause opens a NEW
# predication, whose own subject may head the anaphor's antecedent.
_CLAUSE_BOUNDARY = re.compile(
    r",\s*|;\s*|:\s*|\s+and\s+|\s+but\s+|\s+or\s+|\s+then\s+|"
    r"\s+when\s+|\s+while\s+|\s+because\s+|\s+so\s+|\s+as\s+|"
    r"\s+which\s+|\s+that\s+|\s+who\s+|\s+그리고\s+|\s+하면\s+",
    re.IGNORECASE)

# The frame a clause opens with: a WH-word or a performative head is
# grammar — the clause's content stands after it.
_CLAUSE_FRAME = re.compile(
    r"^(?:please\s+|also\s+|then\s+|and\s+|but\s+)?"
    r"(?:what|why|when|where|who|which|how|that|whether|if)\s+",
    re.IGNORECASE)

# The anaphora classes: what agreement the antecedent must show.
_PLURAL_ANAPHORA = frozenset({
    "they", "them", "these", "those", "we", "us", "their", "ours"})
_SINGULAR_ANAPHORA = frozenset({
    "it", "this", "that", "he", "she", "his", "her", "my", "your"})


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*|[가-힣]+", text or "")


def _head_nominal(clause: str) -> str:
    """The clause's own subject: the nominal that opens it.

    The frame and the closed class are grammar and are skipped; the
    scan stops at the clause's PREDICATION, so what remains is the
    nominal a pronoun in a later clause can point back at."""
    from brain.dialogue_act import _LEADING_PERFORMATIVE
    text = str(clause or "").strip()
    text = _LEADING_PERFORMATIVE.sub("", text, count=1).strip()
    text = _CLAUSE_FRAME.sub("", text).strip()
    predicate = _predicate_words()
    head: list[str] = []
    for token in _words(text):
        low = token.lower().strip("'-")
        if not low or low in _LOCAL_FUNCTION:
            if head:
                break
            continue
        if low in predicate or low.endswith("ing") and len(low) > 5                 and head and low not in predicate:
            break
        head.append(token)
        if len(head) >= 3:
            break
    return " ".join(head).strip(" -")


def _is_plural(nominal: str) -> bool:
    parts = _words(nominal)
    if not parts:
        return False
    if " and " in f" {nominal.lower()} ":
        return True
    last = parts[-1].lower()
    return last.endswith("s") and not last.endswith("ss") \
        and last not in _LOCAL_FUNCTION


_DEMONSTRATIVES = frozenset({"this", "that", "these", "those"})


def _anaphoric_head_noun(text: str, position: int, anaphor: str) -> str:
    """The noun the anaphor itself carries ("these skills" → skills).

    A determiner plus a head noun agrees with ITS OWN head, so what it
    needs of an antecedent is a KIND: the part of the antecedent that
    names the kind, not the quantity the kind is measured in.  A bare
    pronoun ("they", "it") carries no head noun and decides nothing
    here."""
    if str(anaphor or "").lower() not in _DEMONSTRATIVES:
        return ""
    tail = str(text or "")[position + len(anaphor):]
    match = re.match(
        r"\s+([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2})",
        tail)
    if not match:
        return ""
    words = [w for w in _words(match.group(1))
             if w.lower() not in _LOCAL_FUNCTION]
    return " ".join(words[:2]).strip()


def _partitive_complement(clause: str) -> str:
    """The KIND a partitive names: the nominal after its last `of`
    ("years of cooking experience" → cooking experience).

    The nominal that HEADS the partitive measures it (a quantity); a
    determiner+noun anaphor points at the kind.  Structurally the same
    for every partitive — nothing about what it happens to say."""
    matches = list(re.finditer(r"\bof\s+([^,.;:!?]+)", str(clause or ""),
                               re.IGNORECASE))
    if not matches:
        return ""
    words = [w for w in _words(matches[-1].group(1))
             if w.lower() not in _LOCAL_FUNCTION]
    return " ".join(words[:3]).strip()


def _shares_content(nominal: str, head_noun: str) -> bool:
    """Surface agreement between a candidate and the anaphor's own
    head noun — a content word they share, never a stop word."""
    left = {w.lower() for w in _words(nominal)
            if w.lower() not in _LOCAL_FUNCTION}
    right = {w.lower() for w in _words(head_noun)
             if w.lower() not in _LOCAL_FUNCTION}
    return bool(left and right and left & right)


def _head_noun_occurrence(chunk: str, head_noun: str) -> str:
    """The anaphor's own head noun where this chunk states it: the
    nominal that contains it.  A determiner+noun anaphor agrees with
    its head on the SURFACE — the region that names that noun is the
    region that holds its referent."""
    wanted = [w.lower() for w in _words(head_noun)
              if w.lower() not in _LOCAL_FUNCTION]
    if not wanted:
        return ""
    words = _words(chunk)
    for index, word in enumerate(words):
        if word.lower() != wanted[0]:
            continue
        start = index
        while start > 0 and words[start - 1].lower() not in _LOCAL_FUNCTION:
            start -= 1
            if index - start >= 2:
                break
        end = index + 1
        while end < len(words) and words[end].lower() not in _LOCAL_FUNCTION:
            end += 1
            if end - index > 2:
                break
        return " ".join(words[start:end]).strip()
    return ""


def _nominals(chunk: str, head_noun: str = "") -> list[tuple[str, str]]:
    """The chunk's candidates: its subject, the kind its partitive
    names when it has one, and — when the anaphor carries its own head
    noun — the nominal that names that noun here.  Each is tagged with
    the role it plays, so the ranking can read the role."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(nominal: str, flavor: str) -> None:
        nominal = str(nominal or "").strip()
        if not nominal or nominal.lower() in seen:
            return
        seen.add(nominal.lower())
        out.append((nominal, flavor))

    add(_head_nominal(chunk), "subject")
    add(_partitive_complement(chunk), "complement")
    if head_noun:
        add(_head_noun_occurrence(chunk, head_noun), "agreement")
    return out


def _find_anaphor(text_lower: str, anaphor: str) -> int:
    """The anaphor's position as a WHOLE WORD.

    A plain substring search finds "it" inside "Circuit" and "he"
    inside "the": the reference then resolves to the fragment before
    the accidental hit ("Circu"), which becomes the question's own
    subject.  Latin anaphors match with word boundaries; others are
    matched as they are."""
    a = str(anaphor or "").lower()
    if not a:
        return -1
    if a.isascii():
        m = re.search(r"(?<![A-Za-z0-9])" + re.escape(a)
                      + r"(?![A-Za-z0-9])", text_lower)
        return m.start() if m else -1
    return text_lower.find(a)


def _local_antecedent(request: str, anaphor: str) -> tuple[str, str, float] | None:
    """The antecedent the REQUEST itself supplies, in the order the
    discourse is actually read: the clause the anaphor stands in, then
    the rest of its sentence, then the sentences before it, then the
    whole request.  A conversation candidate is never consulted while
    the request still names a referent of its own.

    Within a region the BEST candidate wins: a determiner+noun anaphor
    is served by the candidate that shares its head noun, else by the
    kind a partitive names, else by the clause's subject.  Regions are
    still taken nearest first — proximity outranks any bonus."""
    text = str(request or "")
    if not text or not anaphor:
        return None
    lowered = text.lower()
    position = _find_anaphor(lowered, anaphor)
    if position < 0:
        return None

    sentence_start = max(text.rfind(".", 0, position),
                         text.rfind("!", 0, position),
                         text.rfind("?", 0, position)) + 1
    sentence = text[sentence_start: position + len(anaphor)]
    clause_offset = 0
    boundaries = list(_CLAUSE_BOUNDARY.finditer(sentence))
    cut = -1
    anaphor_in_sentence = position - sentence_start
    for match in boundaries:
        if match.start() <= anaphor_in_sentence:
            cut = match.end()
        else:
            break
    if cut >= 0:
        clause = sentence[cut:]
        clause_offset = cut
    else:
        clause = sentence

    # region → score: nearer structure, higher confidence
    regions = [
        (clause[: anaphor_in_sentence - clause_offset], 0.9, "clause"),
        (sentence[: anaphor_in_sentence], 0.8, "sentence"),
    ]
    before_sentence = text[:sentence_start].strip()
    if before_sentence:
        regions.append((before_sentence, 0.7, "earlier sentence"))
    if position > 0:
        regions.append((text[:position], 0.6, "request"))

    plural = anaphor.lower() in _PLURAL_ANAPHORA
    singular = anaphor.lower() in _SINGULAR_ANAPHORA
    head_noun = _anaphoric_head_noun(text, position, anaphor)
    for region, base, where in regions:
        best: tuple[str, float] | None = None
        for chunk in reversed(_CLAUSE_BOUNDARY.split(region) or []):
            for nominal, flavor in _nominals(chunk, head_noun):
                score = base
                if plural:
                    score += 0.05 if _is_plural(nominal) else -0.2
                elif singular:
                    score -= 0.05 if _is_plural(nominal) else 0.0
                if head_noun:
                    if _shares_content(nominal, head_noun):
                        score += 0.35
                    elif flavor == "complement":
                        # the kind, not the quantity that measures it
                        score += 0.30
                if score < 0.5:
                    continue
                if best is None or score > best[1]:
                    best = (nominal, score)
        if best is not None:
            return (best[0], where, min(0.95, best[1]))
    return None


def _discourse_text(request: str) -> str:
    """The part of the request that can carry a DISCOURSE reference.

    A response-shaping unit ("keep it under two minutes", "make it
    short") refers to the REPLY — its pronoun asks nothing of the
    topic state, so a turn whose only anaphor stands inside one never
    becomes a clarification about the wrong thing."""
    text = str(request or "")
    if not text:
        return text
    try:
        from brain.dialogue_act import analyze_focus
        units = analyze_focus(text).units
    except Exception:
        return text
    spans = [tuple(u["span"]) for u in units
             if u.get("kind") == "constraint"
             and len(u.get("span") or []) == 2]
    if not spans:
        return text
    kept = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            kept.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        kept.append(text[cursor:])
    return " ".join(part for part in kept if part.strip())


def _pair_reference(anaphora) -> str:
    """The pair expression an utterance carries, or ""."""
    for surface in anaphora:
        text = str(surface or "").strip().lower()
        if not text:
            continue
        for pair in _PAIR_REFERENCES:
            if text == pair or text.startswith(pair + " ") or pair in text:
                return str(surface)
    return ""


def _anaphora_in(text: str) -> list[str]:
    """The anaphoric expressions the utterance contains, in order.

    The detection is the ACT LAYER'S grammar (the demonstrative's
    distribution: a determiner over an open-class nominal is deixis,
    a standing demonstrative is a reference), plus the discourse's
    recipient phrases — never a table of which word means which
    referent.
    """
    from brain.dialogue_act import _anaphor_hits, _TOPIC_ANAPHORA_PHRASES
    lower = (text or "").lower()
    if not lower:
        return []
    found = [word for _position, word in _anaphor_hits(lower)]
    found += [phrase for phrase in _TOPIC_ANAPHORA_PHRASES
              if phrase in lower]
    # The pair expression is a referring expression too: it points at
    # the discourse's other member.
    found += [pair for pair in _PAIR_REFERENCES if pair in lower]
    return _dedupe(found)


def _follows_the_reference(topic: str, request: str, anaphora) -> bool:
    """A same-utterance candidate that overlaps or follows the reference
    is not an antecedent: "explain that in a simpler way" has "way"
    after "that", and the whole request ("Can you explain that ...")
    CONTAINS the reference — a referent cannot be its own antecedent."""
    lower = (request or "").lower()
    position = _find_anaphor(lower, str(anaphora[0])) if anaphora else -1
    if position < 0:
        return False
    topic_pos = lower.find(str(topic).lower())
    if topic_pos < 0:
        return False
    return topic_pos + len(str(topic)) > position


def _turn_text(turn) -> str:
    if isinstance(turn, str):
        return turn
    if hasattr(turn, "user_input"):
        return getattr(turn, "user_input", "") or ""
    if isinstance(turn, dict):
        return str(turn.get("user_input", "") or "")
    return ""


# The fields a turn's STRUCTURED record carries: what the turn was
# about (topic, topics), what it carried (entities, concepts), what the
# response SAID (claims, response_representation) and the knowledge it
# was built from (referenced_knowledge).
_STATE_TOPICS = ("topic", "topics")
_STATE_LISTS = ("entities", "concepts", "claims")
_STATE_RECORDS = ("semantic_representation", "response_representation",
                  "referenced_knowledge")


def _turn_state(turn) -> dict:
    """A previous turn's structured meaning — the record the discourse
    keeps, so the next turn's resolution reads WHAT THE TURN MEANT
    instead of re-parsing its raw text."""
    if isinstance(turn, dict):
        state = {k: turn.get(k) for k in
                 _STATE_TOPICS + _STATE_LISTS + _STATE_RECORDS
                 if turn.get(k)}
        return state
    state = {}
    for key in _STATE_TOPICS + _STATE_LISTS + _STATE_RECORDS:
        value = getattr(turn, key, None)
        if isinstance(value, (list, tuple)):
            value = [v for v in value if v]
        if isinstance(value, dict):
            value = value or None
        if value:
            state[key] = value
    return state


def _state_values(state: dict, *keys) -> list[str]:
    out: list[str] = []
    for key in keys:
        value = state.get(key)
        if isinstance(value, str):
            value = [value]
        for item in (value or []):
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _entry_text(item) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("content") or item.get("text")
                   or item.get("sentence") or "").strip()
    return str(getattr(item, "content", "")
               or getattr(item, "text", "") or "").strip()


# The representation's content-bearing roles: what the response SAYS,
# as opposed to how it frames or attributes it.
_CONTENT_ROLES = {"core", "support"}


def _state_claims(state: dict) -> list[str]:
    """What the previous response SAID: the claims as they were
    reasoned (the representation's own record), never a re-parse."""
    claims = _state_values(state, "claims")
    representation = state.get("response_representation") or {}
    if isinstance(representation, dict):
        claims += _state_values(
            {"claims": representation.get("claims")}, "claims")
        for key, role_key in (("moves", "role"),
                              ("sentence_plan", "function")):
            for item in (representation.get(key) or []):
                if isinstance(item, dict) and str(
                        item.get(role_key) or "") not in _CONTENT_ROLES:
                    continue
                if not isinstance(item, dict) and str(
                        getattr(item, role_key, "") or "") \
                        not in _CONTENT_ROLES:
                    continue
                text = _entry_text(item)
                if text and text not in claims:
                    claims.append(text)
    semantic = state.get("semantic_representation") or {}
    if isinstance(semantic, dict):
        claims += _state_values(
            {"claims": semantic.get("claims")}, "claims")
    return _dedupe(claims)


def _state_referent(state: dict) -> tuple[str, str]:
    """The knowledge the previous response was built from: the target
    it answered about (kind, target)."""
    knowledge = state.get("referenced_knowledge") or {}
    if isinstance(knowledge, dict):
        target = str(knowledge.get("target") or "").strip()
        if target:
            return ("topic", target)
    return ("", "")


def _reference_source(semantic) -> str:
    """Where the topic layer recorded the referent came from —
    `utterance`, `discourse_topic`, `previous_turn` or nothing."""
    dialogue_intent = getattr(semantic, "dialogue_intent", None)
    if isinstance(dialogue_intent, dict):
        scope = dialogue_intent.get("scope") or {}
        if isinstance(scope, dict):
            return str(scope.get("reference_source") or "")
    scope = getattr(dialogue_intent, "scope", None)
    if isinstance(scope, dict):
        return str(scope.get("reference_source") or "")
    return ""


def _refers_to(source: str) -> str:
    """The reference's provenance, named for the trace."""
    if source == "topic_state":
        return "previous_topic"
    if source == "previous_turn":
        return "previous_turn"
    if source == "utterance":
        return "utterance"
    return "previous_topic"


def _turn_topics(turn) -> list[str]:
    """The topics recorded for a previous turn (the semantic
    representation the turn carried, when it is available)."""
    if isinstance(turn, dict):
        return [str(t) for t in (turn.get("topics") or []) if t]
    topics = getattr(turn, "topics", None)
    if topics:
        return [str(t) for t in topics if t]
    return []


def _proposition_of(text: str) -> str:
    """A previous utterance as a proposition (its content, framed)."""
    import re
    stripped = re.sub(
        r"^(what|who|whom|which|how|why|when|where)"
        r"(?:\s*'(?:s|re)\b|\s+(?:(?:is|are|was|were|do|does|did|"
        r"can|could|would|will|should|to)\b))?\s*",
        "", text.strip().rstrip("??.!"), flags=re.IGNORECASE)
    return (stripped or text).strip()[:80]


def _same_referent(a: str, b: str) -> bool:
    """Two topical expressions name the same referent when their
    content words agree (articles and framing are grammar)."""
    def content(text: str) -> set[str]:
        return {w for w in
                (w.strip(".,!?()").lower() for w in str(text).split())
                if w and w not in {"a", "an", "the", "of", "in", "for",
                                   "and", "to", "is", "are", "that",
                                   "this"}}
    ca, cb = content(a), content(b)
    if not ca or not cb:
        return False
    return len(ca & cb) / max(1, min(len(ca), len(cb))) >= 0.6
