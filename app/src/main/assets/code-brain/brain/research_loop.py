"""The research loop: generic query reformulation over evidence quality.

When a retrieval comes back empty or thin, the loop REFORMULATES the
query through structural strategies (synonym expansion, concept
decomposition, entity clarification, constraint relaxation, domain
terminology, source refinement) and tries again — bounded by an
attempt count and a wall-clock budget.  No question is looked up and
no canned search term is mapped: the strategies are transforms of the
QUERY's own shape and the turn's semantic context.

    original query → sufficiency check → (failed) → reformulate →
    retrieve → check → … → sufficient / exhausted / budget hit
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TIME_BUDGET = 10.0

STRATEGY_SEQUENCE = (
    "synonym_expansion",
    "request_framing",
    "concept_decomposition",
    "entity_clarification",
    "constraint_relaxation",
    "domain_terminology",
    "source_refinement",
)

_QUERY_FRAMING = frozenset({
    "what", "is", "are", "the", "a", "an", "how", "do", "does", "did",
    "why", "when", "where", "who", "which", "to", "of", "in", "on",
    "for", "with", "and", "or", "please", "tell", "me", "about",
    "explain", "describe", "define", "example", "examples", "step",
    "by", "its", "it", "this", "that", "그", "는", "은", "이", "가",
    "을", "를", "설명", "알려줘", "뭐", "무엇", "예시",
})


@dataclass
class ResearchOutcome:
    """What one research-loop run produced: the query that was used,
    the raw results, the evaluated knowledge (when buildable), and the
    attempts it took — the trace reads the whole history, not just the
    last shot."""
    query: str = ""
    results: list = field(default_factory=list)
    knowledge: object = None
    attempts: int = 0
    queries: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    strategy_used: str = ""
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "attempts": self.attempts,
            "queries": list(self.queries),
            "strategies": list(self.strategies),
            "strategy_used": self.strategy_used,
            "reason": self.reason,
            "results": len(self.results),
        }


def _query_terms(query: str) -> list[str]:
    terms = []
    for word in str(query or "").split():
        stripped = word.strip("'\"`.,;:!?()[]").lower()
        if stripped and stripped not in _QUERY_FRAMING:
            terms.append(stripped)
    return terms


# The words that NAME something rather than describe it: a retriever
# finds an article by a name, and a name never carries the phrase that
# qualifies it.
_QUERY_QUALIFIERS = re.compile(
    r"\b(?:the|a|an|most|more|very|really|key|main|best|top|good|"
    r"important|fundamental|basic|general|various|different|"
    r"entire|whole|entire|subject|topic|thing|aspect|point)\b",
    re.IGNORECASE)

# Where a describing phrase parts from the name it qualifies.
_HEAD_SPLIT = re.compile(
    r"\s+(?:of|from|with|for|in|on|and|that|which|as|over|across|"
    r"through|during|between)\s+", re.IGNORECASE)


def _query_heads(query: str) -> list[str]:
    """The query's own nominal heads, longest first: the phrase a
    speaker used, then the name inside it, then its complement.  Every
    candidate is a TRIM of this query — never a canned term, never a
    topic word chosen for a particular question."""
    text = " ".join(str(query or "").split())
    if not text:
        return []
    out: list[str] = []

    def offer(piece: str) -> None:
        piece = piece.strip(" \t,.;:!?-")
        if len(piece) < 3 or len(piece) > 48:
            return
        lowered = piece.lower()
        if lowered == text.lower():
            return
        if any(lowered == e for e in out):
            return
        out.append(piece)

    trimmed = _QUERY_QUALIFIERS.sub(" ", text)
    offer(trimmed)
    match = _HEAD_SPLIT.search(trimmed)
    if match:
        left = trimmed[:match.start()]
        right = trimmed[match.end():]
        offer(left)
        offer(_QUERY_QUALIFIERS.sub(" ", left))
        if not _HEAD_SPLIT.search(right):
            offer(right)
    while _QUERY_QUALIFIERS.search(text):
        shorter = _QUERY_QUALIFIERS.sub(" ", text)
        if shorter == text:
            break
        offer(shorter)
        text = shorter
    # the shortest NAME first: a retriever matches an article by its
    # name, and the bare head is the name the longer phrase qualifies
    return sorted(out, key=len)


def _requirement_queries(requirement) -> list[str]:
    """The parts the requirement recorded — object or trace dict."""
    if requirement is None:
        return []
    if isinstance(requirement, dict):
        raw = requirement.get("queries") or []
    else:
        raw = getattr(requirement, "queries", None) or []
    return [str(q) for q in raw if str(q).strip()]


def reformulate(original_query: str, context: dict | None = None) -> list[tuple]:
    """The candidate queries, one per structural strategy, in a fixed
    order.  Each candidate is a TRANSFORM of the query and the context
    — never a canned term for a particular question."""
    context = dict(context or {})
    original = str(original_query or "").strip()
    terms = _query_terms(original) or [original.lower()]
    entity = str(context.get("entity") or "").strip()
    topic = str(context.get("topic") or "").strip()
    activated = [str(a) for a in (context.get("activated") or []) if a]
    expand = context.get("expand")
    candidates: list[tuple] = []

    def offer(strategy: str, query: str) -> None:
        query = " ".join(str(query).split())
        if not query or query.lower() == original.lower():
            return
        if any(query.lower() == existing.lower()
               for _, existing in candidates):
            return
        candidates.append((strategy, query))

    for term in sorted(terms, key=len, reverse=True):
        variants = []
        if callable(expand):
            try:
                variants = [str(v) for v in (expand(term) or []) if v]
            except Exception:
                variants = []
        for variant in variants[:2]:
            if variant.lower() == term:
                continue
            expanded = re.sub(re.escape(term), variant, original,
                              count=1, flags=re.IGNORECASE)
            if expanded.strip().lower() == original.strip().lower():
                expanded = f"{original} {variant}"
            offer("synonym_expansion", expanded)
            break

    request_terms = [str(t) for t in (context.get("request_terms") or [])
                     if t]
    if request_terms:
        framed = " ".join(dict.fromkeys(request_terms))[:120]
        offer("request_framing", framed)

    if len(terms) >= 2:
        longest = max(terms, key=len)
        offer("concept_decomposition", longest)

    if entity and entity.lower() not in original.lower():
        offer("entity_clarification", f"{entity} {original}")

    relaxed = [t for t in terms if len(t) > 2]
    if relaxed and len(relaxed) != len(original.split()):
        offer("constraint_relaxation", " ".join(relaxed))

    if activated:
        joined = " ".join(activated[:4])
        if joined.lower() != original.lower():
            offer("domain_terminology", joined)

    if original:
        offer("source_refinement", f"{original} site:wikipedia.org")

    if not candidates and topic and topic.lower() != original.lower():
        offer("entity_clarification", topic)
    return candidates


def research_retrieve(original_query: str, requirement=None, *,
                      retrieve, evaluate, is_sufficient=None,
                      context: dict | None = None,
                      max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                      time_budget: float = DEFAULT_TIME_BUDGET,
                      clock=None) -> ResearchOutcome:
    """Run the bounded research loop: retrieve, evaluate, reformulate,
    repeat until the evidence is sufficient, the attempts run out, or
    the wall clock says stop.  Partial evidence from earlier attempts
    is kept as the honest fallback; an empty outcome means every
    attempt found nothing."""
    clock = clock or time.monotonic
    context = dict(context or {})
    started = clock()
    outcome = ResearchOutcome(query=str(original_query or ""))
    if is_sufficient is None:
        def is_sufficient(knowledge) -> bool:
            try:
                return bool(knowledge and knowledge.claims(limit=1))
            except Exception:
                return False
    candidates = [( "original", str(original_query or ""))]
    # The PARTS the request itself stated are candidate queries: a
    # multi-part ask is retrieved part by part, so each part gets its
    # own evidence instead of the request's own text standing in.
    _original = str(original_query or "").strip().lower()
    for head in _query_heads(original_query):
        if head.lower() != _original:
            candidates.append(("nominal_head", head))
    for part in _requirement_queries(requirement):
        if part.strip().lower() != _original:
            candidates.append(("requested_part", part.strip()))
    candidates.extend(reformulate(original_query, context))
    fallback_results: list = []
    fallback_query = ""
    for strategy, query in candidates:
        if outcome.attempts >= max_attempts:
            outcome.reason = "attempts exhausted"
            break
        if outcome.attempts and clock() - started > time_budget:
            outcome.reason = "time budget exhausted"
            break
        outcome.attempts += 1
        outcome.queries.append(query)
        outcome.strategies.append(strategy)
        try:
            results = list(retrieve(query) or [])
        except Exception:
            results = []
        knowledge = None
        if results:
            try:
                knowledge = evaluate(results, query)
            except Exception:
                knowledge = None
        if knowledge is not None and is_sufficient(knowledge):
            outcome.query = query
            outcome.results = results
            outcome.knowledge = knowledge
            outcome.strategy_used = strategy
            outcome.reason = "sufficient"
            return outcome
        if results and not fallback_results:
            fallback_results = results
            fallback_query = query
        if not results and not outcome.reason:
            outcome.reason = "no results"
    if fallback_results:
        try:
            knowledge = evaluate(fallback_results, fallback_query)
        except Exception:
            knowledge = None
        outcome.query = fallback_query
        outcome.results = fallback_results
        outcome.knowledge = knowledge
        outcome.strategy_used = outcome.strategies[-1] \
            if outcome.strategies else "original"
        if not outcome.reason or outcome.reason == "no results":
            outcome.reason = "exhausted with partial evidence"
        return outcome
    if not outcome.reason:
        outcome.reason = "exhausted"
    outcome.query = str(original_query or "")
    return outcome
