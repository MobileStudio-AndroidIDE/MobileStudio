"""Evidence for a document's knowledge goals.

    Retrieved Evidence -> Evidence Validation -> Evidence Synthesis ->
    Response Representation

Retrieval is driven by the STRUCTURED REQUEST: every query is a
knowledge goal (its subject, its relation, the entities the relation
holds of).  A response requirement never issues a query -- it states
how the reply must read, not a fact about the world.

Validation is what a source must stand up to before any of its text
enters the response: it must be an account of the subject, it must
carry substance, and it must carry what the goals ask about.  A retail
listing for "computers" when the goal is the history of computing, or
a page on English distributives when the goal is learning to program,
is not evidence -- it is dropped with its score and never quoted.
"""
from __future__ import annotations

import re
import time as _time

# A source that SELLS rather than explains: the class is the source's
# own character, never a test of a particular topic.
_COMMERCIAL = re.compile(
    r"(?:/cart|/checkout|/deal|/price|/coupon|/sale|/product[s]?/|"
    r"/store/|/shop/|/category/|\.com/shop|/dp/|/sku)"
    r"|\b(?:best\s?buy|amazon|walmart|newegg|ebay|costco|aliexpress|"
    r"shop\s+now|add\s+to\s+cart|checkout|best\s+price|free\s+shipping|"
    r"price\s+match|deal[s]?\s+on|coupon|discount|on\s+sale|buy\s+now|"
    r"our\s+prices|in\s+stock|out\s+of\s+stock)\b",
    re.IGNORECASE)

# Prose a source must carry before its text can support a claim: a
# title alone is an address, not an account.
_MIN_SUBSTANCE = 120

# What a goal needs of the pool before retrieval may leave it alone:
# a couple of sentences that actually carry its ask.
_GOAL_MIN_CHARS = 700
_POOL_MIN = 2000

_STOP = frozenset((
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "it", "its", "this", "that", "these", "those", "they", "them",
    "their", "we", "you", "your", "our", "i", "me", "my", "what",
    "which", "when", "where", "who", "how", "why", "than", "then",
    "there", "here", "not", "no", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "do", "does", "did",
    "have", "has", "had", "also", "into", "over", "under", "about",
    "more", "most", "some", "any", "all", "each", "every", "such",
    "one", "two", "three", "own", "same", "other", "between", "during",
    "before", "after", "while", "because", "if", "whether", "been",
    "being", "than", "very", "many", "much", "need", "uses", "help",
))


def _stems(text: str) -> set[str]:
    """Four letters is where a word still names itself: it holds
    'star' with 'stars' and 'computing' with 'computer' without ever
    conflating two different subjects.  Sources are held to this."""
    out: set[str] = set()
    for word in re.findall(r"[A-Za-z가-힣][A-Za-z가-힣'’+#-]*",
                           str(text or "").lower()):
        if word in _STOP or len(word) < 4:
            continue
        out.add(word[:4])
    return out


def _keys(text: str) -> set[str]:
    """A passage is ranked more generously than a source is judged:
    a request that asks how things EVOLVED must find the page that
    says EVOLUTION, and 'star' must meet 'stars'."""
    out: set[str] = set()
    for word in re.findall(r"[A-Za-z가-힣][A-Za-z가-힣'’+#-]*",
                           str(text or "").lower()):
        if word in _STOP or len(word) < 4:
            continue
        out.add(word[:4])
        if len(word) >= 6:
            out.add(word[:3])
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    out = []
    for part in parts:
        part = re.sub(r"\s+", " ", part).strip()
        if len(part) > 40:
            out.append(part)
    return out


def _heading_of(section) -> str:
    if isinstance(section, dict):
        return str(section.get("heading") or
                   section.get("title") or "")
    return str(getattr(section, "title", "") or "")


# The wiki markup a source's extract carries: its own heading markers
# and its internal links are formatting, never prose to quote.
_WIKI_MARKUP = re.compile(
    r"\s*={2,6}\s*[^=\n]{1,80}\s*={2,6}\s*|\[\[[^\]]+\]\]")


def _text_of(section) -> str:
    if isinstance(section, dict):
        body = str(section.get("text") or section.get("snippet") or "")
    else:
        body = str(getattr(section, "snippet", "") or "")
    return _WIKI_MARKUP.sub(" ", body)


def _url_of(section) -> str:
    if isinstance(section, dict):
        url = str(section.get("url") or "")
        if not url and section.get("page_title"):
            page = str(section["page_title"]).replace(" ", "_")
            url = "https://en.wikipedia.org/wiki/" + _quote(page)
        return url
    return str(getattr(section, "url", "") or "")


def _quote(page: str) -> str:
    import urllib.parse
    return urllib.parse.quote(page, safe="_()")


def _source_of(section) -> str:
    if isinstance(section, dict):
        return str(section.get("source") or "wikipedia")
    return str(getattr(section, "source", "") or "wikipedia")


class DocumentEvidence:
    """The validated evidence a document's retrieval produced.

    ``material`` holds every validated sentence in the order its
    source stated it; ``by_goal`` maps a knowledge goal to the
    positions of the sentences that carry it, best evidence first, so
    the response can select the strongest and then read them in the
    source's own order."""
    __slots__ = ("queries", "sources", "material", "by_goal", "rejected",
                 "attempts", "fallbacks", "section_of", "section_headings",
                 "order_by_goal")

    def __init__(self):
        self.queries: list[str] = []
        self.sources: list[dict] = []
        self.material: list[str] = []
        self.by_goal: dict[str, list[int]] = {}
        self.rejected: int = 0
        self.attempts: int = 0
        self.fallbacks: int = 0
        self.section_of: list[int] = []
        self.section_headings: list[str] = []
        self.order_by_goal: dict[str, list[int]] = {}

    def to_dict(self) -> dict:
        return {
            "queries": list(self.queries),
            "sources": [dict(s) for s in self.sources],
            "material": list(self.material),
            "by_goal": {k: list(v) for k, v in self.by_goal.items()},
            "rejected": int(self.rejected),
            "attempts": int(self.attempts),
            "fallbacks": int(self.fallbacks),
        }

    def section_heading(self, index: int) -> str:
        if 0 <= index < len(self.section_of):
            position = self.section_of[index]
            if 0 <= position < len(self.section_headings):
                return self.section_headings[position]
        return ""


def validate(section, content_terms: set[str],
             aspect_terms: set[str], subject_only: bool = False) -> bool:
    """Whether a retrieved source may stand as evidence.

    Structural gates, in order: not a transaction page, carrying
    substance, and answering the question rather than only naming its
    subject.  A page that passes on the topic's name and on nothing
    the goals ask about answers a different question."""
    text = f"{_heading_of(section)} {_text_of(section)} {_url_of(section)}"
    if _COMMERCIAL.search(text):
        return False
    body = _text_of(section)
    if len(body.strip()) < _MIN_SUBSTANCE:
        return False
    if subject_only:
        # the subject's own source: a section of the article already
        # accepted for this document carries the account it belongs to,
        # whether or not it repeats the subject's name
        return True
    shared = _stems(body) & content_terms
    if not shared:
        return False
    if shared & aspect_terms or len(shared) >= 2:
        return True
    return bool(_stems(_heading_of(section)) & content_terms)


def _rank(topic_terms: set[str], relation_terms: set[str],
          heading: str, text: str) -> int:
    """How well a passage carries a goal.

    The relation the goal asks about is what makes the passage answer
    it; the subject's name alone only places it near the topic — and a
    passage that names its subject many times still says nothing about
    what was asked, so the subject's contribution is capped.  A source
    heading that the goal asked for is evidence the source is already
    organised the way the request is."""
    words = _keys(text)
    hit = len(words & relation_terms) * 4 + min(len(words & topic_terms), 2)
    if heading and (_keys(heading) & (topic_terms | relation_terms)):
        hit += 4
    return hit


def _subject_article(query: str) -> list:
    """The subject's own source addressed through a query.

    The index resolves the address, the article carries the account: a
    request states its subject the way a speaker states it, and the
    article's own name is not the request's wording."""
    from brain.web_search import _disambiguation_title
    from brain.web_search import wikipedia_search
    from brain.web_search import wikipedia_sections
    try:
        titles = wikipedia_search(query, limit=3) or []
    except Exception:
        return []
    for title in titles:
        try:
            if _disambiguation_title(title):
                continue
            sections = wikipedia_sections(title, max_chars=40000) or []
        except Exception:
            sections = []
        if sections:
            return list(sections)
    return []


def gather(structured, language: str = "en",
           time_budget: float = 20.0, needed: int = 0) -> DocumentEvidence:
    """Retrieve, validate and synthesise evidence for every knowledge
    goal of the document.

    The document's own subject addresses the source first.  Each goal
    then reads the source's own sections and keeps the ones that carry
    what that goal asked for.  A goal the source does not reach falls
    back to the validated pool: the plan stays whole, and no part of
    the response ever announces a hole."""
    from brain.web_search import retrieve as _retrieve
    from brain.web_search import wikipedia_sections

    evidence = DocumentEvidence()
    deadline = _time.time() + time_budget

    content_terms: set[str] = set()
    aspect_terms: set[str] = set()
    goal_topic: dict[str, set[str]] = {}
    goal_relation: dict[str, set[str]] = {}
    topic_keys: set[str] = _keys(structured.topic)
    topic_terms: set[str] = _stems(structured.topic)
    for goal in structured.knowledge_goals:
        goal_topic[goal.id] = _keys(goal.topic) | topic_keys
        goal_relation[goal.id] = _keys(" ".join(goal.relations))
        content_terms |= (_stems(goal.topic) | _stems(goal.required_evidence)
                          | topic_terms)
        aspect_terms |= _stems(" ".join(goal.relations))
    if not content_terms:
        evidence.fallbacks += 1
        return evidence

    pool: list[tuple[str, str, str]] = []          # heading, text, url
    seen: set[str] = set()
    topic_sections: list = []

    def take(sections, subject_only: bool = False) -> None:
        for section in sections:
            body = _text_of(section)
            if not body.strip():
                continue
            if not validate(section, content_terms, aspect_terms,
                            subject_only):
                evidence.rejected += 1
                continue
            key = re.sub(r"\W+", "", f"{_heading_of(section)}{body[:120]}"
                         ).lower()
            if key in seen:
                continue
            seen.add(key)
            pool.append((_heading_of(section), body, _url_of(section)))
            url = _url_of(section)
            if url and not any(s["source"] == url for s in evidence.sources):
                evidence.sources.append({
                    "source": url,
                    "title": _heading_of(section) or structured.topic,
                    "type": _source_of(section),
                })

    def sync() -> list[str]:
        """Flatten the pool into sentences, remembering which source
        section stated each one so a section of the response can read
        as the source read it."""
        out: list[str] = []
        positions: list[int] = []
        used: set[str] = set()
        evidence.section_headings = [h for h, _text, _url in pool]
        for position, (_heading, text, _url) in enumerate(pool):
            for sentence in _sentences(text):
                key = " ".join(sentence.lower().split())
                if key in used:
                    continue
                used.add(key)
                out.append(sentence)
                positions.append(position)
        evidence.material = out
        evidence.section_of = positions
        return out

    def covered(sents: list[str], goal) -> int:
        """How much of a passage already speaks to a goal, counted in
        characters of the sentences that actually carry its ask."""
        top = goal_topic.get(goal.id) or set()
        rel = goal_relation.get(goal.id) or set()
        total = 0
        for sentence in sents:
            if _rank(top, rel, "", sentence):
                total += len(sentence)
        return total

    # 1. the document's own subject addresses the source
    topic = str(structured.topic or "").strip()
    if topic:
        evidence.queries.append(topic)
        evidence.attempts += 1
        try:
            sections = wikipedia_sections(topic, max_chars=40000) or []
        except Exception:
            sections = []
        if not sections:
            # a subject stated the way a speaker states it addresses no
            # article by that exact name; Wikipedia's own index resolves
            # the address while the query stays the request's meaning
            try:
                from brain.web_search import _disambiguation_title
                from brain.web_search import wikipedia_search
                for title in wikipedia_search(topic, limit=3):
                    if _disambiguation_title(title):
                        continue
                    sections = wikipedia_sections(
                        title, max_chars=40000) or []
                    if sections:
                        break
            except Exception:
                sections = []
        if not sections:
            # a search result is a page about the subject, not the
            # subject's own source, so it never earns the relaxed read
            try:
                sections = [{"heading": r.title, "text": r.snippet,
                             "url": r.url, "source": r.source}
                            for r in _retrieve(topic, deep=True)]
            except Exception:
                sections = []
        else:
            topic_sections = list(sections)
        take(sections)

    # 2. a goal the source does not reach addresses it itself; the
    #    plan is kept whole by retrieval rather than by an apology
    pool_material = sync()
    for goal in structured.knowledge_goals:
        if _time.time() > deadline:
            evidence.fallbacks += 1
            continue
        if covered(pool_material, goal) >= _GOAL_MIN_CHARS:
            continue
        query = (goal.topic or goal.question or "").strip()
        if not query or query.lower() == topic.lower():
            evidence.fallbacks += 1
            continue
        evidence.queries.append(query)
        evidence.attempts += 1
        try:
            take([{"heading": r.title, "text": r.snippet, "url": r.url,
                   "source": r.source}
                  for r in _retrieve(query, deep=True)])
        except Exception:
            evidence.fallbacks += 1
        pool_material = sync()

    # 2b. a pool too thin to carry the response reads on: the subject's
    #     own sections the first pass had no need of, then the request's
    #     relations addressed to the source again
    target = max(_POOL_MIN, int(needed or 0))
    candidates: list[str] = []
    for goal in structured.knowledge_goals:
        for relation in goal.relations[:2]:
            query = f"{goal.topic or topic} {relation}".strip()
            if (query and query.lower() != topic.lower()
                    and query.lower() not in {c.lower() for c in candidates}):
                candidates.append(query)
    pending = list(topic_sections)
    retried = False
    while sum(len(s) for s in sync()) < target:
        if _time.time() > deadline:
            evidence.fallbacks += 1
            break
        batch = None
        subject_only = False
        if pending:
            # the subject's own source: its sections the first pass had
            # no goal to place, read for their substance
            batch, pending = pending[:2], pending[2:]
            subject_only = True
        elif candidates:
            query = candidates.pop(0)
            evidence.queries.append(query)
            evidence.attempts += 1
            pending = _subject_article(query)
            if pending:
                continue
            try:
                batch = [{"heading": r.title, "text": r.snippet,
                          "url": r.url, "source": r.source}
                         for r in _retrieve(query, deep=True)]
            except Exception:
                evidence.fallbacks += 1
        elif not retried:
            # the subject addressed once more: an index that refused
            # the first address does not decide what may be said
            retried = True
            evidence.attempts += 1
            pending = _subject_article(topic)
            if pending:
                continue
        if batch is None:
            break
        take(batch, subject_only)
        sync()

    # 3. synthesise: every sentence is available to every goal, best
    #    evidence first, and the goal's own source section is read
    #    before any other so the response's part reads as one account
    sync()
    for goal in structured.knowledge_goals:
        top = goal_topic.get(goal.id) or set()
        rel = goal_relation.get(goal.id) or set()
        scored = [(_rank(top, rel, evidence.section_heading(index),
                         sentence), index)
                  for index, sentence in enumerate(evidence.material)]
        scored = [(rank, index) for rank, index in scored if rank]
        if not scored:
            evidence.fallbacks += 1
            evidence.by_goal[goal.id] = []
            continue
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        evidence.by_goal[goal.id] = [index for _rank_value, index in scored]
        # the source's own section the goal reads best comes first, so
        # one part of the response stays one account instead of a
        # scatter of the article's strongest lines
        strongest: dict[int, int] = {}
        for rank, index in scored:
            position = evidence.section_of[index]                 if index < len(evidence.section_of) else -1
            if position < 0:
                continue
            if rank > strongest.get(position, -1):
                strongest[position] = rank
        evidence.order_by_goal[goal.id] = sorted(
            strongest, key=lambda position: (-strongest[position],
                                             position))

    if not evidence.sources:
        evidence.fallbacks += 1
    return evidence
