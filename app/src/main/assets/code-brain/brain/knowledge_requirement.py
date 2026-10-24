"""Knowledge requirement analysis: does this turn need the WEB?

    Need external / current information?
        YES → Web Search / Knowledge Retrieval → Evidence
        NO  → Self Knowledge / Memory / Context

The decision is made from the turn's SEMANTICS — the act, the resolved
reference, the topic state and the temporal framing of the request —
before any retrieval happens, and it is recorded so the trace shows
whether the web was used and WHY.  Nothing here is a sentence→answer
map: a temporal marker raises a freshness requirement, a style reframe
over an already-resolved topic does not, a self-referential act never
leaves the system.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# The acts answered from the system's own state (identity, capability)
_SELF_ACTS = frozenset({
    "identity", "capability_query", "language_capability_query",
})

# The acts that re-present knowledge the discourse already holds: the
# request changes the REGISTER, not the fact base.
_REFRAMING_ACTS = frozenset({
    "simplify_explanation", "explanation_request",
})

# The acts whose answer is conversational structure (no retrieval).
_CONTEXT_ONLY_ACTS = frozenset({
    "greeting", "conversation", "clarification",
})

# Recency markers: the request is about something CURRENT, so a stored
# (possibly stale) answer is not enough.  This is a temporal class of
# the utterance, never a topic keyword.
_RECENT = (
    r"\b(latest|most recent|recent|current|now|today|yesterday|"
    r"this (?:morning|afternoon|evening|night|week|month|year)|"
    r"just (?:announced|released|happened|posted)|breaking|newest)\b",
    r"\b(20[0-9]{2})\b",
    # a SUPERLATIVE temporal frame: "the last/previous X" is the most
    # recent occurrence of it — a point in time no timeless
    # definition carries ("When was the last World Cup held?").
    r"\b(?:last|previous|most\s+recent)\s+(?:[a-z]+\s+){0,3}"
    r"(?:held|occurred|happened|took\s+place|played|conducted|"
    r"published|released|announced|updated|edition|version|release|"
    r"update|announcement|election|tournament|event|match|game|"
    r"summit|conference|survey|census)\b",
    r"(최신|오늘|어제|지금|방금|최근|요즘|이번\s*(?:주|달|연도))",
)

# A time SCOPE constrains the options ("this weekend"), it does not
# demand fresh information.  Kept separate so "What could I learn this
# weekend?" is a scoped recommendation, not a news query.
_TIME_SCOPE = (
    r"\b(this|next|coming)\s+(?:weekend|week|month|monday|friday|"
    r"summer|holiday)\b",
    r"\bthis (?:saturday|sunday)\b",
    r"(이번\s*(?:주말|주|토|일)|다음\s*(?:주|주말))",
)


@dataclass
class KnowledgeRequirement:
    """The turn's knowledge needs, decided before retrieval."""
    need_web: bool = False
    need_self: bool = False
    need_context: bool = False
    need_local: bool = False
    reason: str = ""
    signals: list[str] = field(default_factory=list)
    freshness: str = "timeless"     # timeless | recent
    time_scope: str = ""
    query: str = ""
    # One query per PART the request itself stated: a multi-part ask
    # is retrieved part by part, never as the request's own text.
    queries: list[str] = field(default_factory=list)

    @property
    def used_web(self) -> bool:
        return self.need_web

    def to_dict(self) -> dict:
        return {
            "need_web": self.need_web,
            "need_self": self.need_self,
            "need_context": self.need_context,
            "need_local": self.need_local,
            "reason": self.reason,
            "signals": list(self.signals),
            "freshness": self.freshness,
            "time_scope": self.time_scope,
            "query": self.query,
            "queries": list(self.queries),
        }


def _resolved(reference) -> bool:
    """Whether the turn's reference resolved (object or serialized form)."""
    if reference is None:
        return False
    if isinstance(reference, dict):
        return bool(reference.get("target")) and not bool(
            reference.get("needs_clarification"))
    return bool(getattr(reference, "resolved", False))


# The intents whose answer is KNOWLEDGE: a content act, an advice ask
# or an investigation.  Their breadth (how many parts the request
# states) decides whether one local concept can cover them.
_KNOWLEDGE_INTENTS = frozenset({
    "information_query", "explanation_request", "simplify_explanation",
    "recommendation_request", "procedural_instruction", "research",
    "command_query", "example_request", "theory",
})


def _matches(patterns, text: str) -> list[str]:
    import re
    hits = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            hits.append(m.group(0).strip().lower())
    return hits


def analyze_knowledge_requirement(semantic, request: str,
                                  reference=None, topic_state=None,
                                  local_available: bool = False,
                                  target: str = "",
                                  intent: str = "",
                                  instance: str = "") -> KnowledgeRequirement:
    """Decide what knowledge THIS turn requires.

    `local_available` is the outcome of the LOCAL knowledge probe (the
    concept store): the requirement distinguishes "already known here"
    from "needs the web" instead of always hitting the network.

    `instance` is the noun of an INSTANCE ask ("example", "예시"): the
    request wants an occurrence of its topic, not its definition, and
    the retrieval is framed as the instance the speaker asked for.
    """
    req = KnowledgeRequirement()
    # the EFFECTIVE act: the act layer's refinement (an unresolved
    # reference turns the turn into a clarification) decides the
    # requirement — the representation's first guess does not.
    intent = str(intent or ((getattr(semantic, "dialogue_intent", None)
                             or {}).get("type")) or "")
    text = request or ""
    req.signals.append(f"intent={intent or 'unknown'}")

    # a requested time scope is recorded (it shapes the options)
    scope_hits = _matches(_TIME_SCOPE, text)
    if scope_hits:
        req.time_scope = scope_hits[0]
        req.signals.append(f"time_scope={req.time_scope}")

    recent_hits = _matches(_RECENT, text)
    if recent_hits:
        req.freshness = "recent"
        req.signals.append(f"recency={recent_hits[0]}")

    if intent in _SELF_ACTS:
        req.need_self = True
        req.need_context = True
        req.reason = "the act is about the system itself: self knowledge"
        req.signals.append("self-referential act")
        return req

    if intent in _CONTEXT_ONLY_ACTS and not _resolved(reference):
        req.need_context = True
        req.reason = "conversational act: conversation context supplies " \
                     "the answer"
        return req

    # An INSTANCE ask: the speaker wants an OCCURRENCE of the topic (an
    # example), not its definition.  The requirement frames the
    # retrieval as the instance the request asked for, wherever an
    # example would come from — a definition never stands in for it.
    if instance:
        req.need_local = bool(local_available)
        req.need_context = True
        req.need_web = True
        req.query = f"{(target or '').strip()} {instance}".strip()
        req.reason = ("the request asks for an instance of the topic: "
                      "an example is sought where one exists")
        req.signals.append(f"instance={instance}")
        return req

    # An INVESTIGATION: the knowledge is not in the conversation yet,
    # so the sources are consulted and the evidence is iterated until
    # it stands up — a stored entry may seed the synthesis but never
    # stands in for the finding-out the request asked for.
    if intent == "research":
        base = (target or "").strip() or text.strip()[:80]
        req.need_web = True
        req.need_local = bool(local_available)
        req.need_context = True
        req.query = base
        req.reason = ("an investigation: external evidence is required "
                      "and iterated until it is sufficient")
        req.signals.append("investigation act")
        return req

    # The request asks for MORE THAN ONE THING: coordinated asks, a
    # scope with two poles, a course through its subject.  Local
    # knowledge of ONE concept cannot cover a multi-part ask, so each
    # part carries its own query and the breadth itself requires
    # evidence.
    if intent in _KNOWLEDGE_INTENTS:
        from brain.longform import progression_frame, request_aspects
        parts = request_aspects(text)
        course = progression_frame(text)
        if len(parts) >= 2 or course:
            base = (target or "").strip() or text.strip()[:80]
            # a part that is essentially the WHOLE request is the
            # request again, not a query: the referent is the query
            parts = [p for p in parts
                     if len(p) < max(60, int(0.6 * len(text)))]
            req.queries = list(parts[:4]) or [base]
            req.need_web = True
            req.need_local = bool(local_available)
            req.need_context = True
            req.query = base
            if parts:
                req.reason = (f"the request asks for {len(parts)} parts: "
                              "each part needs its own evidence")
            else:
                req.reason = ("the request frames a course through its "
                              "subject: one local concept cannot cover it")
            req.signals.append(f"breadth={max(1, len(parts))}")
            return req

    # A style reframe over a resolved reference: the discourse already
    # holds the content — a web search would fetch the WRONG thing (the
    # surface form, not the referent).
    if intent in _REFRAMING_ACTS and _resolved(reference) and \
            local_available:
        req.need_context = True
        req.need_local = True
        req.reason = "a reframe over the resolved topic: context and " \
                     "local knowledge suffice"
        req.signals.append("reference resolved to a known topic")
        return req

    # Freshness demands the web even when a stored answer exists: a
    # stale definition is not an answer to a "latest" question.  This
    # holds for EVERY knowledge act — a procedure, a recommendation and
    # a definition are equally stale when the request is about
    # something current.
    if req.freshness == "recent" and intent in (
            "information_query", "explanation_request",
            "simplify_explanation", "recommendation_request",
            "procedural_instruction", "command_query"):
        req.need_web = True
        req.need_local = False
        req.reason = "the request asks about something current: " \
                     "fresh external evidence is required"
        base = (target or "").strip() or text.strip()[:80]
        scope = " ".join(x for x in (req.time_scope, recent_hits[0])
                         if x and x.lower() not in base.lower()).strip()
        req.query = f"{base} {scope}".strip() if scope else base
        req.signals.append("freshness overrides local evidence")
        return req

    if local_available:
        req.need_local = True
        req.need_context = True
        req.reason = "local knowledge covers the target"
        return req

    if not local_available:
        # A non-content referent must never become a query (a deictic
        # or bare interrogative would fetch noise).
        from brain.conversation_knowledge import ConversationKnowledgeEngine
        if not ConversationKnowledgeEngine._topic_contentful(target):
            req.need_context = True
            req.reason = "the referent is not contentful: retrieval " \
                         "would answer about the wrong thing"
            return req
        req.need_web = True
        req.reason = "no local evidence for the target: external " \
                     "evidence is required"
        req.query = (target or "").strip()
        return req

    req.need_context = True
    req.reason = "conversation context supplies the answer"
    return req
