"""Retrieved knowledge: the evidence a retrieval produced, with its
provenance and its standing.

A retrieval never becomes the answer.  It becomes
`RetrievedKnowledge` — a set of `KnowledgeEvidence` items, each with
its source, its relevant claims and its relevance / consistency /
quality / freshness — which the REASONING ranks before any of it
reaches a response.

    Retrieved Evidence → Relevance → Consistency → Source Quality →
    Freshness → Reasoning → Response Representation
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

# Source quality: what a source is worth as evidence BEFORE its content
# is read (an encyclopedia article vs. an anonymous page).
_SOURCE_QUALITY = {
    "wikipedia": 0.85,
    "web": 0.55,
    "concept_store": 0.8,
    "self_knowledge": 0.9,
    "conversation": 0.7,
}


@dataclass
class KnowledgeEvidence:
    """One item of retrieved evidence, with its provenance."""
    source: str = ""                 # the origin (url / store id / "self")
    source_type: str = ""            # wikipedia | web | concept | self
    title: str = ""
    content: str = ""                # the retrieved content (evidence)
    retrieved_at: float = 0.0
    query: str = ""
    relevant_claims: list[str] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)
    contradicts: list[str] = field(default_factory=list)
    relevance: float = 0.0
    confidence: float = 0.0
    freshness: float = 0.0
    quality: float = 0.0
    consistency: float = 0.0
    # how the evidence stood up against the other sources: several
    # sources agree / one source alone / the sources disagree
    verification_status: str = ""

    def __post_init__(self):
        if not self.retrieved_at:
            self.retrieved_at = time.time()

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.retrieved_at)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "content": _preview(self.content),
            "retrieved_at": self.retrieved_at,
            "query": self.query,
            "relevant_claims": list(self.relevant_claims),
            "supports": list(self.supports),
            "contradicts": list(self.contradicts),
            "relevance": round(self.relevance, 3),
            "confidence": round(self.confidence, 3),
            "freshness": round(self.freshness, 3),
            "quality": round(self.quality, 3),
            "consistency": round(self.consistency, 3),
            "verification_status": self.verification_status,
        }


@dataclass
class RetrievedKnowledge:
    """Everything a retrieval produced for one requirement."""
    query: str = ""
    requirement: dict = field(default_factory=dict)
    evidence: list[KnowledgeEvidence] = field(default_factory=list)
    retrieved_at: float = 0.0
    web_used: bool = False
    reason: str = ""
    # how many retrieved items the entity gate rejected (a page not
    # about the queried entity is not evidence, however it scored)
    gated: int = 0
    # the research loop's history: every query it tried and how many
    # attempts the evidence took to stand up
    queries: list[str] = field(default_factory=list)
    attempts: int = 1
    # the language the response is being built in: the claims composed
    # from this evidence are framed in THAT language's register
    language: str = "en"

    def __post_init__(self):
        if not self.retrieved_at:
            self.retrieved_at = time.time()

    def ranked(self) -> list[KnowledgeEvidence]:
        """Evidence ordered by the reasoning's standing: relevance,
        then consistency, then source quality, then freshness."""
        return sorted(
            self.evidence,
            key=lambda e: (-e.relevance, -e.consistency, -e.quality,
                           -e.freshness))

    @property
    def best(self) -> KnowledgeEvidence | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def claims(self, limit: int = 3) -> list[str]:
        out: list[str] = []
        for evidence in self.ranked():
            for claim in evidence.relevant_claims:
                if claim and claim not in out:
                    out.append(claim)
                if len(out) >= limit:
                    return out
        return out

    def sources(self) -> list[str]:
        return [e.source for e in self.ranked() if e.source]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "web_used": self.web_used,
            "reason": self.reason,
            "requirement": dict(self.requirement),
            "retrieved_at": self.retrieved_at,
            "language": self.language,
            "gated": self.gated,
            "queries": list(self.queries),
            "attempts": self.attempts,
            "evidence": [e.to_dict() for e in self.ranked()],
        }


def _preview(text: str, limit: int = 200) -> str:
    text = _strip_citations(text)
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


# Citation markers a source leaves in its own text: reference numbers,
# "[cite]", "[citation needed]".  They are the SOURCE'S apparatus, not
# its content — they never reach a claim, a title or a response.
_CITATION = re.compile(
    r"\s*\[(?:\d{1,3}|[a-z]{1,3}|cite\s*\d*|citation(?:\s+needed)?|"
    r"verification\s+needed|footnote\s*\d*|nb\s*\d*)\]", re.IGNORECASE)


def _strip_citations(text: str) -> str:
    """The source's own text without its citation apparatus."""
    out = _CITATION.sub("", text or "")
    return re.sub(r"\s{2,}", " ", out).strip()


def _tokens(text: str) -> set[str]:
    return {w for w in (w.strip(".,!?;:'\"()[]").lower()
                        for w in (text or "").split())
            if len(w) > 2 and w not in _STOP}


_STOP = {
    "the", "and", "for", "that", "this", "with", "from", "are", "was",
    "were", "have", "has", "can", "could", "would", "should", "their",
    "what", "which", "when", "where", "how", "why", "you", "your",
}

# A procedural question's claims that announce a procedure come first:
# the sentence that says HOW is the one the ask waited for.
_CLAIM_CUE_RE = re.compile(
    r"(?:how\s+(?:to|can|do)\b|step[- ]by[- ]step|\bguide\b|"
    r"\btutorial\b|방법|가이드|튜토리얼|만들)", re.IGNORECASE)


def _relevance(content: str, query: str) -> float:
    a, b = _tokens(content), _tokens(query)
    if not a or not b:
        return 0.3
    overlap = len(a & b) / len(b)
    return max(0.1, min(1.0, 0.35 + 0.65 * overlap))


def _split_claims(content: str) -> list[str]:
    import re as _re
    stripped = _strip_citations(content)
    parts = [p.strip() for p in _re.split(r"(?<=[.!?])\s+", stripped)]
    # A source that drops the period between its own sentences: the
    # boundary is still a boundary — a lowercase word followed by a
    # sentence-initial capital carrying a verb or article.  It is only
    # cut when what precedes it is a full statement, so a name inside
    # a clause is never treated as a sentence start.
    boundary = _re.compile(
        r"(?<=[a-z])\s+(?=[A-Z][a-z]{2,}\b(?:\s+\w+){0,2}\s+"
        r"(?:is|are|was|were|has|have|had|can|could|will|would|do"
        r"|does|did|don|doesn|didn|the|a|an|in|on|of|used|with)\b)")
    expanded: list[str] = []
    for part in parts:
        cursor = 0
        start = 0
        while True:
            match = boundary.search(part, cursor)
            if not match:
                break
            head = part[start:match.start()].strip()
            if len(head) >= 40 or _looks_like_heading(head):
                expanded.append(head)
                start = match.end()
            cursor = match.end()
        tail = part[start:].strip()
        if tail:
            expanded.append(tail)
    # A heading the source set before its text ("Finding Black Holes"
    # + the sentence): the heading names, the sentence states — the
    # claim is the statement, and only when the heading actually has
    # a statement behind it.
    trimmed = []
    for index, item in enumerate(expanded):
        if _looks_like_heading(item) and index + 1 < len(expanded):
            continue
        trimmed.append(item)
    return [p for p in trimmed if len(p) > 20]


def _looks_like_heading(text: str) -> bool:
    words = [w for w in (text or "").split() if w[:1].isalpha()]
    if not words or len(text) > 40 or len(words) > 6:
        return False
    capped = sum(1 for w in words if w[:1].isupper())
    return capped >= len(words) - 1


def _ordered_content_words(text: str) -> list[str]:
    """The text's content words IN ORDER — the head is the last one."""
    return [w for w in (w.strip(".,!?;:'\"()[]").lower()
                        for w in (text or "").split())
            if len(w) > 2 and w not in _STOP]


def entity_gate(query: str, title: str, content: str) -> bool:
    """Whether the page is ABOUT the queried entity.

    Two structural readings decide it, never a topic test: the page's
    TITLE names one of the query's content words, or the query's head
    word appears in the page's own text.  A page that satisfies neither
    matched no entity — it is not evidence, however relevant it scored.
    """
    ordered = _ordered_content_words(query)
    if not ordered:
        return True
    head = ordered[-1]
    title_words = _tokens(title)
    if title_words and (title_words & set(ordered)):
        return True
    return head in _tokens(content) or head in title_words


_ANAPHOR_LEAD = re.compile(
    r"(?i)^(it|this|that|these|those|they|he|she|we)\b")


def claim_gate(claim: str, query: str, title: str,
               first: bool = False) -> bool:
    """Whether a claim speaks about the entity: it names the query's
    words, leads with an anaphor the page's subject anchors (a page
    about the entity has nothing else for its opening pronoun), or it
    is the page's OPENING statement — the sentence written directly
    under the title stands under that title by the page's own form.
    A line sharing none of these is boilerplate — it never becomes
    knowledge."""
    claim_words = _tokens(claim)
    if not claim_words:
        return False
    if first:
        return True
    if _ANAPHOR_LEAD.match((claim or "").strip()):
        return True
    anchor = _tokens(title) | _tokens(query)
    if not anchor:
        return True
    return bool(claim_words & anchor)


def build_from_web(results, query: str, requirement=None,
                   question: str = "") -> RetrievedKnowledge:
    """Turn raw web results into ranked evidence (never into answers).

    `question` is the request's own words: relevance is scored against
    the topic AND the question, so a guide that answers "how do I make
    an android app" outranks an encyclopedia article that only shares
    the topic's name."""
    knowledge = RetrievedKnowledge(
        query=query,
        requirement=requirement.to_dict() if hasattr(requirement, "to_dict")
        else (dict(requirement) if isinstance(requirement, dict) else {}),
        web_used=True,
        reason=getattr(requirement, "reason", "") or "",
    )
    scoring_query = " ".join(
        part for part in (str(query or ""), str(question or "")) if part)
    now = time.time()
    for item in results or []:
        title = _strip_citations(getattr(item, "title", "") or "")
        url = getattr(item, "url", "") or ""
        snippet = _strip_citations(getattr(item, "snippet", "") or "")
        source_type = getattr(item, "source", "") or "web"
        # The entity gate: a page not about the queried entity never
        # enters the evidence — its score measures nothing.
        if not entity_gate(query, title, snippet):
            knowledge.gated += 1
            continue
        claims = []
        for index, claim in enumerate(_split_claims(snippet)):
            if claim_gate(claim, query, title, first=(index == 0)):
                claims.append(claim)
        if len(claims) > 1 and _CLAIM_CUE_RE.search(scoring_query):
            cued_claims = [c for c in claims if _CLAIM_CUE_RE.search(c)]
            if cued_claims:
                claims = cued_claims + [c for c in claims
                                        if c not in cued_claims]
        evidence = KnowledgeEvidence(
            source=url or title,
            source_type=source_type,
            title=title,
            content=snippet,
            retrieved_at=now,
            query=query,
            relevant_claims=claims,
            relevance=_relevance(snippet + " " + title, scoring_query),
            quality=_SOURCE_QUALITY.get(source_type, 0.5),
            # a freshly retrieved page is as fresh as a retrieval gets
            freshness=1.0 if source_type in _SOURCE_QUALITY else 0.7,
        )
        knowledge.evidence.append(evidence)

    # consistency: how much the sources AGREE — an item that shares its
    # content words with the others is corroborated, an outlier is not.
    for evidence in knowledge.evidence:
        mine = _tokens(evidence.content)
        if not mine or len(knowledge.evidence) == 1:
            evidence.consistency = 1.0 if mine else 0.5
            continue
        agree = 0
        for other in knowledge.evidence:
            if other is evidence:
                continue
            theirs = _tokens(other.content)
            if mine & theirs:
                agree += 1
        evidence.consistency = min(1.0, agree / max(1, len(knowledge.evidence) - 1))
        evidence.confidence = round(
            0.5 * evidence.relevance + 0.3 * evidence.quality
            + 0.2 * evidence.consistency, 3)
    count = len(knowledge.evidence)
    for evidence in knowledge.evidence:
        if not evidence.content.strip() or not evidence.relevant_claims:
            evidence.verification_status = "insufficient"
        elif count <= 1:
            evidence.verification_status = "single_source"
        elif evidence.consistency >= 0.5:
            evidence.verification_status = "corroborated"
        elif evidence.consistency <= 0.0:
            evidence.verification_status = "conflicting"
        else:
            evidence.verification_status = "single_source"
    return knowledge


def build_from_store(concept_hits, query: str, requirement=None) -> RetrievedKnowledge:
    """Local concept-store evidence in the same representation, so the
    reasoning treats both sources identically."""
    knowledge = RetrievedKnowledge(
        query=query,
        requirement=requirement.to_dict() if hasattr(requirement, "to_dict")
        else (dict(requirement) if isinstance(requirement, dict) else {}),
        web_used=False,
        reason=getattr(requirement, "reason", "") or "local knowledge",
    )
    for concept, score in list(concept_hits or []):
        description = getattr(concept, "description", "") or ""
        evidence = KnowledgeEvidence(
            source=getattr(concept, "id", "") or "",
            source_type="concept",
            title=getattr(concept, "name", "") or "",
            content=description,
            query=query,
            relevant_claims=_split_claims(description),
            relevance=min(1.0, float(score or 0.0) / 8.0),
            quality=_SOURCE_QUALITY["concept_store"],
            freshness=1.0,
            confidence=min(1.0, 0.5 + float(score or 0.0) / 16.0),
        )
        knowledge.evidence.append(evidence)
    return knowledge
