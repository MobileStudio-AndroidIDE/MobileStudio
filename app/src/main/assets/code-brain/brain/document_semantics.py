"""Document-level semantic representation.

A long request is ONE document, analysed before anything is retrieved:

    the request → Document Semantic Analysis → Local Reference
    Resolution → Structured Request (topic, knowledge goals,
    response requirements) → Knowledge Requirements → Retrieval →
    Evidence Validation → Knowledge Synthesis → Response Plan →
    Long-form Generation → Verification

The two things a document carries are never confused with each
other.  Knowledge goals are the facts the speaker wants established —
each with its own subject, its own asked-for relation and its place
in the dependency chain the request states.  Response requirements are
how the reply must be shaped — its depth, its organisation, its
examples, its audience.  A response requirement asks for no knowledge
of its own and never reaches a retriever: "clear sections" is a demand
on the layout, never a query about the world.

Every division here is read from the request's own grammar — its
coordination of clauses, its performative frame, its anaphora, and the
closed classes of words that shape a reply.  No sentence is mapped to
an answer and no benchmark rule is encoded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ── Clause structure ────────────────────────────────────────────────
# A clause boundary is where a NEW act begins: after punctuation when
# what follows is itself a request (a wh-word, a knowledge act, an
# additive), or after a colon-like comma before a performative.  The
# split is grammar — the shape of the coordination — never the words a
# particular subject happens to use.
_CLAUSE_BOUNDARY = re.compile(
    r"(?<=[,;])\s*(?=(?:and\s+|or\s+|then\s+|also\s+)?"
    r"(?:how|what|why|when|where|who|which|whether|also|including|"
    r"first|next|then|finally|note|say|add|make\s+sure|keep|avoid|"
    r"ensure|cover|describe|explain|tell|show|list|discuss|define|"
    r"give|provide|include|compare|contrast|summar\w*|"
    r"알려|설명|보여|정리|비교|요약|나열)\b)"
    r"|(?<=[.!?。！？])\s+",
    re.IGNORECASE)

_LEADING_CONDUCTOR = re.compile(
    r"^(?:and\s+|or\s+|then\s+|also\s+|plus\s+|but\s+|now\s+|"
    r"finally\s+|next\s+|first\s+|including\s+|include\s+|"
    r"with\s+|also,?\s+|더불어|그리고|또한|그리고\s+)", re.IGNORECASE)

# The performative head of a clause: the act the speaker performs over
# its complement.  Stripping it leaves what the act is ABOUT.
_PERFORMATIVE = re.compile(
    r"^(?:(?:please\s+|plz\s+|pls\s+|just\s+|also\s+|then\s+)?"
    r"(?:(?:can|could|would|will|shall)\s+(?:you\s+)?|would\s+you\s+mind\s+)?"
    r"(?:explain\w*|describ\w*|tell(?:\s+me)?|show(?:\s+me)?|"
    r"give(?:\s+me)?|provide(?:\s+me)?|offer(?:\s+me)?|share\w*|send(?:\s+me)?|"
    r"present\w*|demonstrate\w*|illustrate\w*|define|list|compare|contrast|"
    r"summar\w*|analy\w*|discuss\w*|clarif\w*|cover|enumerat\w*|teach|"
    r"walk\s+me\s+through|break\s+down|say|mention|describe|note|add|"
    r"research\w*|investigate\w*|explore|review|examine|survey|stud(?:y|ies)|"
    r"알려|보여|설명|비교|정리|요약|나열|가르쳐)"
    r"(?:\s+me)?)(?:\s+)", re.IGNORECASE)

_WH = re.compile(r"^(how|what|why|when|where|who|which|whose|whether)\b",
                 re.IGNORECASE)


# ── Lexical classes (closed classes, extended as the grammar is) ────
# A reply nominal names the communicative product the addressee is to
# produce.  Its presence with a governing act or a shaping modifier
# makes the clause a demand on the RESPONSE, not a question about the
# world.
_REPLY_NOMINAL = re.compile(
    r"\b(?:explanations?|descriptions?|answers?|summaries|summary|"
    r"guides?|reports?|overviews?|outlines?|responses?|walkthroughs?|"
    r"tutorials?|articles?|essays?|breakdowns?|briefings?|"
    r"introductions?|lessons?|plans?|roadmaps?|curricula|curriculum|"
    r"syllabi|syllabus|schedules?|programmes?|profiles?|notes?|"
    r"write-up|writeup|drafts?)\b|"
    r"설명서|가이드|계획|요약|답변|정리", re.IGNORECASE)

_GOVERNING_ACT = re.compile(
    r"\b(?:i\s+(?:want|wish|need|would\s+like|prefer|am\s+looking\s+for)|"
    r"want|wish|would\s+like|prefer|looking\s+for|"
    r"please\s+(?:give|provide|write|offer|share))\b", re.IGNORECASE)

_GOVERNING_CLAUSE = re.compile(
    r"(?:^|(?<=[.!?]\s))(?:give|provide|offer|share|send|present|write|"
    r"draw\s+up|put\s+together|keep|ensure|make\s+sure|aim\s+for|"
    r"maintain|deliver|produce)\b", re.IGNORECASE)

_SHAPE_ADJECTIVE = re.compile(
    r"\b(?:detailed|clear|structured|well[- ]?"
    r"structured|short|brief|concise|comprehensive|thorough|"
    r"step[- ]by[- ]step|simple|easy|understandable|readable|accessible|"
    r"organized|organised|coherent|practical|complete|naive|"
    r"beginner[- ]friendly|beginner|overview|introductory|"
    r"명확한|자세한|체계적|구조화된|간단한|쉬운|이해하기\s*쉽|초보자용)\b",
    re.IGNORECASE)

# How the reply must be shaped: the closed set of response-requirement
# kinds a request can state.  Phrases are matched, never sentences.
_REQUIREMENT_KINDS: tuple[tuple[str, re.Pattern], ...] = (
    ("depth", re.compile(
        r"\b(?:in\s+(?:more\s+)?detail|in\s+depth|in\s+full|at\s+length|"
        r"thoroughly|comprehensively|fully|extensively|deeply|detailed|"
        r"detailled|elaborate|step[- ]by[- ]step|"
        r"자세하게|상세하게|상세히|구체적으로|단계별로)\b",
        re.IGNORECASE)),
    ("organization", re.compile(
        r"\b(?:clear\s+sections?|sections?|structured|structure\s+it|"
        r"structured\s+plan|outline|organized|organised|"
        r"organized\s+into|sectioned|break\s+it\s+down|"
        r"ordering|order\s+it|in\s+order|stage[- ]by[- ]stage|"
        r"구조화|구조|단계|나누어|나눠)\b", re.IGNORECASE)),
    ("examples", re.compile(
        r"\b(?:examples?|instances?|sample\s+cases?|worked\s+examples?|"
        r"concrete\s+cases?)\b|예시|예제|사례|보기", re.IGNORECASE)),
    ("audience", re.compile(
        r"\b(?:for\s+a\s+beginner|beginners?|beginner[- ]friendly|"
        r"almost\s+no\s+experience|no\s+experience|starting\s+from\s+no|"
        r"newcomers?|novices?|layperson|lay\s+people|general\s+audience|"
        r"understandable|in\s+simple\s+terms|in\s+plain\s+terms|"
        r"easy\s+to\s+follow|accessible|"
        r"초보자|입문자|초심자|이해하기\s*쉽게|쉽게)\b", re.IGNORECASE)),
    ("length", re.compile(
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|"
        r"thirty|forty|fifty|hundred|\d+)\s*"
        r"(?:minutes?|seconds?|hours?|days?|weeks?|months?|words?|"
        r"sentences?|lines?|paragraphs?|pages?|characters?|steps?)\b",
        re.IGNORECASE)),
    ("caution", re.compile(
        r"\b(?:common\s+mistakes?|mistakes?|pitfalls?|gotchas?|"
        r"cautions?|failures|what\s+to\s+avoid|traps?)\b|"
        r"실수|함정|주의", re.IGNORECASE)),
)

# How a clause that is not wh-initial states its subject: the scope
# prepositions that bound the noun phrase the request is about.
_SCOPE_PREP = re.compile(
    r"\s+(?:from|over|through|during|within|across|throughout|between|"
    r"without|under|after|before|along|via|per|toward|towards)\b",
    re.IGNORECASE)

# The grammatical auxiliary: a clause cut at it leaves its subject.
_AUX = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "can", "could", "may", "might", "shall",
    "should", "must", "gets", "get", "got",
}

# The finite/light predicate of an embedded clause: cutting before it
# leaves the NP the predicate is about.
_PREDICATE = {
    "change", "changes", "changed", "changing",
    "work", "works", "worked", "working",
    "happen", "happens", "happened",
    "differ", "differs", "differed",
    "vary", "varies", "varied",
    "grow", "grows", "grew", "grown",
    "develop", "develops", "developed",
    "evolve", "evolves", "evolved",
    "become", "becomes", "became",
    "make", "makes", "made", "causes", "cause", "caused",
    "create", "creates", "created",
    "produce", "produces", "produced",
    "matter", "matters", "mattered",
    "influence", "influences", "influenced",
    "depend", "depends", "depended",
    "require", "requires", "required",
    "consist", "consists", "consisted",
    "exist", "exists", "existed",
    "remain", "remains", "remained",
    "lead", "leads", "led",
    "die", "dies", "died", "born", "formed", "formation",
    "move", "moves", "moved", "turn", "turns", "turned",
    "shift", "shifts", "shifted", "flow", "flows", "flowed",
    "use", "uses", "used", "convert", "converts", "converted",
    "store", "stores", "stored", "release", "releases", "released",
    "harvest", "harvests", "harvested", "fix", "fixes", "fixed",
    "need", "needs", "needed", "want", "wants", "wanted",
    "run", "runs", "ran", "build", "builds", "built",
    "write", "writes", "wrote", "read", "reads", "read",
    "learn", "learns", "learned", "learn",
    "practice", "practices", "practiced",
    "know", "knows", "knew", "understand", "understands",
    "fail", "fails", "failed", "succeed", "succeeds",
    "start", "starts", "started", "finish", "finishes",
    "improve", "improves", "improved", "progress", "progresses",
    "stall", "stalls", "stalled", "review", "reviews", "reviewed",
    "evaluate", "evaluates", "evaluated",
}

_SUBORDINATOR = re.compile(
    r"\b(?:when|while|because|since|after|before|unless|whereas|"
    r"if|that|whether|as)\s+", re.IGNORECASE)

# The pronouns and light definites a later clause points back with.
_ANAPHOR = re.compile(
    r"\b(?:it|its|they|them|their|this|that|these|those|he|she|his|"
    r"her|one|ones)\b", re.IGNORECASE)
_LIGHT_DEFINITE = re.compile(
    r"\b(?:the|this|that|these|those)\s+"
    r"(?:process|processes|system|stage|stages|step|steps|plan|plans|"
    r"way|ways|case|cases|situation|matter|aspect|parts?|"
    r"change|changes|model|structure|approach|method)\b", re.IGNORECASE)

_CONTENT_WORD = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣'’+#-]*|\d+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
    "been", "being", "it", "its", "this", "that", "these", "those",
    "they", "them", "their", "we", "you", "your", "our", "i", "me", "my",
    "what", "which", "when", "where", "who", "how", "why", "than", "then",
    "there", "here", "not", "no", "can", "could", "would", "should",
    "will", "shall", "may", "might", "must", "do", "does", "did", "done",
    "have", "has", "had", "also", "into", "over", "under", "about",
    "more", "most", "some", "any", "all", "each", "every", "such",
    "one", "two", "three", "own", "same", "other", "between", "during",
    "before", "after", "while", "because", "if", "whether", "include",
    "including", "including", "please", "just", "very", "much", "many",
    "get", "got", "make", "made", "take", "taken", "give", "given",
}


def _words(text: str) -> list[str]:
    return [w.lower() for w in _CONTENT_WORD.findall(str(text or ""))]


def _content_words(text: str) -> list[str]:
    return [w for w in _words(text) if w not in _STOP and len(w) > 1]


def _phrase(words: list[str], limit: int = 6) -> str:
    return " ".join(words[:limit])


# ── The structures ──────────────────────────────────────────────────

@dataclass
class KnowledgeGoal:
    """One unit of knowledge the document asks to have established.

    `topic` is what the goal is about, `question` the relation it asks
    for, `relations` the predicate side of that ask, and `depends_on`
    the goal whose answer this one builds on — the document's own
    dependency chain, in the order the speaker stated it."""
    id: str = ""
    order: int = 0
    topic: str = ""
    question: str = ""
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    confidence: float = 0.0
    clause: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "order": self.order,
            "topic": self.topic,
            "question": self.question,
            "entities": list(self.entities),
            "relations": list(self.relations),
            "depends_on": list(self.depends_on),
            "required_evidence": list(self.required_evidence),
            "queries": list(self.queries),
            "confidence": round(self.confidence, 3),
            "clause": self.clause,
        }


@dataclass
class ResponseRequirement:
    """How the reply must be shaped.  Never content, never a query."""
    kind: str = ""
    value: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value,
                "source": self.source}


@dataclass
class StructuredRequest:
    """The document's structure: its subject, the knowledge goals it
    asks for with their dependencies, the requirements it places on
    the response, and the constraints the speaker put on what is
    being planned."""
    topic: str = ""
    topic_source: str = ""
    knowledge_goals: list[KnowledgeGoal] = field(default_factory=list)
    response_requirements: list[ResponseRequirement] = field(
        default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    clauses: list[dict] = field(default_factory=list)
    references: list[dict] = field(default_factory=list)

    @property
    def queries(self) -> list[str]:
        """Every retrieval query this document may issue — built only
        from the knowledge goals.  A response requirement can never
        appear here by construction."""
        out: list[str] = []
        for goal in self.knowledge_goals:
            for query in goal.queries:
                if query and query not in out:
                    out.append(query)
        return out

    def requirements_of(self, kind: str) -> list[ResponseRequirement]:
        return [r for r in self.response_requirements if r.kind == kind]

    def has_requirement(self, kind: str) -> bool:
        return bool(self.requirements_of(kind))

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "topic_source": self.topic_source,
            "knowledge_goals": [g.to_dict() for g in self.knowledge_goals],
            "response_requirements": [r.to_dict()
                                      for r in self.response_requirements],
            "constraints": [dict(c) for c in self.constraints],
            "clauses": [dict(c) for c in self.clauses],
            "references": [dict(r) for r in self.references],
            "queries": list(self.queries),
        }


# ── Segmentation ────────────────────────────────────────────────────

def split_clauses(request: str) -> list[dict]:
    """The document's clauses WITH their spans, from its own
    coordination.  A clause boundary is where a new act begins; a
    blank line separates paragraphs, whose topics are the last scope
    before the whole request."""
    text = str(request or "").strip()
    if not text:
        return []
    pieces = [p for p in _CLAUSE_BOUNDARY.split(text) if p and p.strip()]
    clauses: list[dict] = []
    breaks = [m.start() for m in re.finditer(r"\n\s*\n", text)]
    cursor = 0
    for piece in pieces:
        raw = piece.strip()
        start = text.find(raw, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(raw)
        paragraph = sum(1 for b in breaks if b < start)
        clauses.append({"text": raw, "span": [start, cursor],
                        "paragraph": paragraph})
    return clauses


def _sentence_index(clauses: list[dict]) -> list[int]:
    """Which sentence each clause belongs to."""
    out: list[int] = []
    index = 0
    previous_end = ""
    for clause in clauses:
        text = clause["text"]
        if previous_end and previous_end[-1:] in ".!?。！？":
            index += 1
        out.append(index)
        previous_end = text
    return out


# ── Clause reading ──────────────────────────────────────────────────

def _strip_performative(clause: str) -> tuple[str, bool]:
    """The clause without its act verb: the complement the act is
    over.  Returns (complement, had_performative)."""
    text = clause.strip()
    match = _PERFORMATIVE.match(text)
    if match:
        return text[match.end():].strip(), True
    return text, False


def _reply_complement(clause: str) -> str:
    """The content the reply nominal carries: "a detailed guide TO
    learning computer programming" is about learning computer
    programming — the prepositional complement names the subject the
    reply must cover."""
    match = _REPLY_NOMINAL.search(clause)
    if not match:
        return ""
    tail = clause[match.end():]
    complement = re.match(
        r"^\s*(?:to|for|about|on|regarding|covering|over|through|of)\s+"
        r"(.+)$", tail, re.IGNORECASE)
    if not complement:
        return ""
    text = _cut_scope(complement.group(1))
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(" \t,.;:!?-")[:120]


def _is_response_shaping(clause: str) -> bool:
    """Whether the clause states a demand on the RESPONSE rather than
    a question about the world.

    Structural: the clause names a reply nominal AND is governed by a
    speech/desire act or modified by a shaping adjective, or it is a
    response-management predicate.  "Clear sections and examples" is
    layout; "how stars are born" is knowledge."""
    text = _LEADING_CONDUCTOR.sub("", clause.strip()).strip()
    if not text:
        return False
    from brain.dialogue_act import _RESPONSE_MANAGEMENT
    if _RESPONSE_MANAGEMENT.match(text):
        return True
    nominal = _REPLY_NOMINAL.search(text)
    if not nominal:
        return False
    if _SHAPE_ADJECTIVE.search(text) or _GOVERNING_ACT.search(text) \
            or _GOVERNING_CLAUSE.search(text):
        return True
    # a bare reply nominal as the clause's own head: what remains after
    # it is a modifier of the reply, never a predication about a topic
    head = _words(text[:nominal.start()])
    return not [w for w in head if w not in _STOP]


_FIRST_PERSON_SUBJECT = re.compile(
    r"^(?:i|we|my|our|i['’]?m|i['’]?ve|"
    r"저는|나는|제가|우리는|내가|내)\b", re.IGNORECASE)

_QUANTITY_RESOURCE = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"twenty|thirty|forty|fifty|hundred)\s*[-\s]?"
    r"(?:(?:minutes?|seconds?|hours?|days?|weeks?|months?|years?|"
    r"dollars?)\b|시간|분|일|주|월|년|원)",
    re.IGNORECASE)

_AVAILABILITY_VERB = re.compile(
    r"(?:(?:have|has|had|need|needs|want|can|cannot|can['’]?t|"
    r"could|spare|spend|afford|allow|limited|limit|lack|only)\b"
    r"|없습니다|없어|없는|없다|필요합니다|필요해|필요한|필요가|"
    r"가진|가져|있어|있는|여유가|시간이)",
    re.IGNORECASE)

_REQUEST_HEAD = re.compile(
    r"\b(?:plan|guide|schedule|routine|list|answer|reply|response|"
    r"explanation|overview|summary|outline|break|session|workout|"
    r"itinerary|itineraries|menu|recipe)\b|"
    r"일정|계획|가이드|목록|설명|개요|답",
    re.IGNORECASE)


def _is_speaker_constraint(clause: str) -> bool:
    """A clause stating how much the SPEAKER has or can spend is not
    something to look up — it constrains what is being PLANNED.

    The distinction is the clause's own predicate: the time a person
    possesses ("I only have 30 minutes a day") is a fact about the
    plan, while the time they REQUEST something for ("I want a
    30-minute plan") is the head noun of the ask itself.  Reading the
    first as a knowledge goal sends the retrieval off to whatever
    happens to mention the same words — which is how an availability
    statement ends up quoted back as if it were the subject."""
    text, _had = _strip_performative(str(clause or ""))
    text = _LEADING_CONDUCTOR.sub("", text).strip()
    if not _FIRST_PERSON_SUBJECT.match(text):
        return False
    quantity = _QUANTITY_RESOURCE.search(text)
    if not quantity:
        return False
    if _REQUEST_HEAD.search(text[quantity.end():quantity.end() + 40]):
        return False
    return bool(_AVAILABILITY_VERB.search(text))


def _response_requirements(clause: str) -> list[ResponseRequirement]:
    """The response-shaping demands this clause states."""
    out: list[ResponseRequirement] = []
    text = clause.strip()
    for kind, pattern in _REQUIREMENT_KINDS:
        match = pattern.search(text)
        if match:
            out.append(ResponseRequirement(
                kind=kind,
                value=re.sub(r"\s+", " ", match.group(0)).strip(),
                source=text[:160]))
    return out


def _strip_shape_phrases(text: str) -> str:
    """The clause without its response-shaping phrases: what the
    request asks FOR never carries how it asked for it.  This is what
    keeps "in detail" or "with clear sections" out of a retrieval
    query."""
    out = str(text or "")
    for _kind, pattern in _REQUIREMENT_KINDS:
        out = pattern.sub(" ", out)
    out = _EXTENT_SHAPE.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip(" \t,.;:!?-")


_EXTENT_SHAPE = re.compile(
    r"(?:,?\s*\b(?:in\s+(?:more\s+)?detail|in\s+depth|in\s+full|"
    r"at\s+length|thoroughly|comprehensively|fully|briefly|concisely|"
    r"shortly|quickly|simply|plainly|carefully|extensively|"
    r"in\s+simple\s+terms|in\s+plain\s+terms|step\s+by\s+step|"
    r"step-by-step|from\s+start\s+to\s+finish|end\s+to\s+end)\b|"
    r",?\s*(?:자세하게|상세하게|상세히|간단히|간결하게|단계별로|"
    r"구체적으로|쉽게|짧게))", re.IGNORECASE)


def _topic_from_clause(clause: str) -> str:
    """What the clause is ABOUT: the noun phrase its predicate is
    predicated of, or the noun phrase itself when it states no
    predicate."""
    text, _had = _strip_performative(clause)
    text = _LEADING_CONDUCTOR.sub("", text).strip()
    text = re.sub(r"^(?:please\s+|just\s+|also\s+)", "", text,
                  flags=re.IGNORECASE).strip()
    wh = bool(_WH.match(text))
    if wh:
        text = _WH.sub("", text, count=1).strip()
        text = re.sub(r"^(?:do|does|did|is|are|was|were|can|could|will|"
                      r"would|should|may|might)\s+(?:it\s+|they\s+|we\s+)?",
                      "", text, flags=re.IGNORECASE).strip()
        wh_body = text
        first = _words(wh_body[:24])
        if first and (first[0] in _PREDICATE or first[0] in _AUX):
            # the wh-word stood in for the subject: the clause states
            # its own subject after a subordinator ("what happens WHEN
            # massive stars die") or straight after its light predicate
            after = _SUBORDINATOR.search(wh_body)
            if after:
                wh_body = wh_body[after.end():].strip()
            else:
                opened = re.match(r"^[A-Za-z가-힣']+\s+", wh_body)
                if opened:
                    wh_body = wh_body[opened.end():].strip()
                wh_body = re.sub(
                    r"^(?:during|within|through|across|under|over|"
                    r"between|without|into|onto|by|on|at|in|for|with|"
                    r"from)\s+", "", wh_body, flags=re.IGNORECASE).strip()
        cut = _cut_predicate(wh_body)
        if _content_words(cut) == _content_words(wh_body):
            cut = _cut_scope(wh_body)
        text = cut
    else:
        text = _cut_scope(text)
    text = re.sub(r"\s+", " ", text).strip(" \t,.;:!?-")
    text = re.sub(r"^to\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(?:a|an|the)\s+", "", text, flags=re.IGNORECASE).strip()
    return text[:120]


def _cut_predicate(text: str) -> str:
    """Cut at the clause's finite predicate, leaving its subject.

    An auxiliary present in the clause IS its finite verb ("stars ARE
    born"); otherwise the last predicate token is ("each stage
    INFLUENCED the next one"), since a noun that also names a process
    ("the major technological changes HAPPENED") is subject, not
    predicate, only when something finite follows it."""
    auxiliary = None
    last = None
    for match in re.finditer(r"[A-Za-z가-힣']+", text):
        if match.start() == 0:
            continue
        word = match.group(0).lower()
        if word in _AUX and auxiliary is None:
            auxiliary = match.start()
        if word in _PREDICATE:
            last = match.start()
    cut = auxiliary if auxiliary is not None else last
    if cut is None:
        return text
    head = text[:cut].strip(" \t,.;:")
    return head if _content_words(head) else text


def _cut_scope(text: str) -> str:
    """Cut at the scope preposition that bounds the noun phrase."""
    match = _SCOPE_PREP.search(text)
    if match and match.start() > 3:
        head = text[:match.start()].strip(" \t,.;:")
        if _content_words(head):
            return head
    return text


# ── Local reference resolution ──────────────────────────────────────

def _resolve_anaphor(text: str, scopes: list[list[str]]) -> tuple[str, str]:
    """Bind the clause's referring expression against the scopes in
    order: the current clause, the current sentence, the current
    paragraph, the whole request, the conversation.  The referent the
    nearest scope supplies wins; the sentence's own main topic is what
    a pronoun in that sentence points at."""
    if not (_ANAPHOR.search(text) or _LIGHT_DEFINITE.search(text)):
        return text, ""
    for scope in scopes:
        for candidate in scope:
            if candidate:
                return text, candidate
    return text, ""


def _bind(text: str, referent: str) -> str:
    if not referent:
        return text
    out = text
    out = re.sub(r"\b(?:they|them|their|these|those)\b", referent, out,
                 count=1, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:it|its|this|that)\b", referent, out, count=1,
                 flags=re.IGNORECASE)
    out = re.sub(_LIGHT_DEFINITE, referent, out, count=1)
    return re.sub(r"\s+", " ", out).strip()


# ── The analysis ────────────────────────────────────────────────────

def analyze_document(request: str, semantic=None,
                     conversation_topics: list[str] | None = None
                     ) -> StructuredRequest:
    """Analyse the whole request as ONE document.

    Segmentation → clause reading (content ask vs response shaping) →
    local reference resolution → the structured request.  The topic is
    the subject of the first knowledge goal (or the content the reply
    nominal carries); the knowledge goals keep the speaker's order and
    each depends on the one before it; the response requirements are
    recorded apart, where no retriever can reach them."""
    structured = StructuredRequest()
    raw = str(request or "").strip()
    if not raw:
        return structured

    clauses = split_clauses(raw)
    sentences = _sentence_index(clauses)

    goals: list[KnowledgeGoal] = []
    request_scope: list[str] = []
    paragraph_scope: list[str] = []
    sentence_scope: list[str] = []
    sentence_topic = ""
    previous_sentence = -1
    previous_paragraph = -1

    for index, clause in enumerate(clauses):
        text = clause["text"]
        paragraph = int(clause.get("paragraph") or 0)
        if paragraph != previous_paragraph:
            if previous_paragraph >= 0:
                request_scope = request_scope + paragraph_scope
                paragraph_scope = []
            previous_paragraph = paragraph
        if sentences[index] != previous_sentence:
            if previous_sentence >= 0:
                sentence_scope = []
            sentence_topic = ""
            previous_sentence = sentences[index]

        stripped = _LEADING_CONDUCTOR.sub("", text).strip()
        body, _had = _strip_performative(stripped)
        body_clean = _LEADING_CONDUCTOR.sub("", body).strip()
        shaping = _is_response_shaping(text)

        # What the speaker HAS or can SPEND constrains the plan, so it
        # is recorded apart from both: a constraint is never a subject
        # for retrieval and never a demand on how the reply is shaped.
        quantity = _QUANTITY_RESOURCE.search(text)
        if quantity and _is_speaker_constraint(text):
            structured.constraints.append({
                "kind": "availability",
                "value": re.sub(r"\s+", " ", quantity.group(0)).strip(),
                "source": text[:200],
            })
            structured.clauses.append({
                "text": text[:200], "kind": "constraint",
                "topic": quantity.group(0),
            })
            continue

        # The response requirements are read from EVERY clause: an
        # extent adverbial inside a content act ("works IN DETAIL")
        # shapes the reply exactly as a separate shaping clause does.
        # They are recorded apart, where no retriever can reach them.
        for requirement in _response_requirements(text):
            if not any(r.kind == requirement.kind
                       and r.value == requirement.value
                       for r in structured.response_requirements):
                structured.response_requirements.append(requirement)

        # 1. local reference resolution: this clause's own referent
        #    first, then the sentence's main topic, then back through
        #    the document and the conversation.
        scopes = [[sentence_scope[0]] if sentence_scope else [],
                  list(reversed(sentence_scope)),
                  list(reversed(paragraph_scope)),
                  list(reversed(request_scope)),
                  list(conversation_topics or [])]
        _text, referent = _resolve_anaphor(body_clean, scopes)
        bound = _bind(body_clean, referent) if referent else body_clean
        clause_topic = _topic_from_clause(bound)
        if not clause_topic:
            clause_topic = _topic_from_clause(text)

        if shaping:
            content = _reply_complement(text)
            if content:
                if not structured.topic:
                    structured.topic = content
                    structured.topic_source = "reply nominal"
                request_scope.append(content)
                # The content the reply must carry is a knowledge goal
                # in its own right: the document is ABOUT it even when
                # no clause predicates something of it directly.
                goal = KnowledgeGoal(
                    id=f"g{len(goals) + 1}",
                    order=len(goals),
                    topic=content,
                    question=content,
                    entities=_content_words(content),
                    relations=[],
                    depends_on=[goals[-1].id] if goals else [],
                    confidence=0.7,
                    clause=text[:200],
                )
                goal.required_evidence = _phrase(goal.entities, 10)
                goal.queries = _goal_queries(goal)
                goals.append(goal)
                sentence_scope.append(content)
                paragraph_scope.append(content)
                structured.clauses.append({
                    "text": text[:200], "kind": "reply_subject",
                    "topic": content, "goal": goal.id,
                })
                continue
            structured.clauses.append({
                "text": text[:200], "kind": "response_shaping",
                "topic": content,
            })
            continue

        if not clause_topic or not _content_words(clause_topic):
            structured.clauses.append({
                "text": text[:200], "kind": "skipped", "topic": clause_topic})
            continue

        asked = _strip_shape_phrases(body_clean)
        question = re.sub(r"\s+", " ", asked or body_clean).strip(" \t,.;:")
        entities = _content_words(clause_topic)
        relation_words = [w for w in _content_words(question)
                          if w not in set(entities)]
        goal = KnowledgeGoal(
            id=f"g{len(goals) + 1}",
            order=len(goals),
            topic=clause_topic,
            question=question or clause_topic,
            entities=entities,
            relations=relation_words,
            depends_on=[goals[-1].id] if goals else [],
            confidence=0.9 if _WH.match(body_clean) or entities else 0.6,
            clause=text[:200],
        )
        goal.required_evidence = _phrase(
            list(dict.fromkeys(entities + relation_words)), 10)
        goal.queries = _goal_queries(goal)
        goals.append(goal)

        if referent:
            structured.references.append({
                "clause": text[:160], "expression": referent,
                "resolved_to": referent,
            })
        if not sentence_scope:
            sentence_topic = clause_topic
        sentence_scope.append(clause_topic)
        paragraph_scope.append(clause_topic)
        request_scope.append(clause_topic)
        structured.clauses.append({
            "text": text[:200], "kind": "knowledge", "topic": clause_topic,
            "goal": goal.id,
        })

    structured.knowledge_goals = goals
    if not structured.topic and goals:
        structured.topic = goals[0].topic
        structured.topic_source = "first knowledge goal"
    if not structured.topic:
        topics = list(getattr(semantic, "intent_topics", None) or [])
        if topics:
            structured.topic = str(topics[0]).strip()
            structured.topic_source = "dialogue intent"
    if not structured.topic:
        goal = getattr(semantic, "goal", None)
        candidate = str(getattr(goal, "target", "") or "").strip()
        if candidate and _content_words(candidate):
            structured.topic = candidate[:120]
            structured.topic_source = "goal target"
    return structured


def _goal_queries(goal: KnowledgeGoal) -> list[str]:
    """The queries this goal may issue: its subject, the relation it
    asks for and the entities that relation holds of — never a
    response requirement, which states no fact about the world."""
    topic = " ".join(goal.entities[:6]).strip()
    if not topic:
        return []
    queries = [topic]
    relation = " ".join(goal.relations[:6]).strip()
    if relation:
        candidate = f"{topic} {relation}"
        if candidate.lower() != topic.lower():
            queries.append(candidate[:120])
    return list(dict.fromkeys(q for q in queries if q))


def knowledge_requirements(structured: StructuredRequest) -> list[dict]:
    """The document's knowledge requirements as SEMANTIC OBJECTS —
    each carrying what must be asked, what the evidence must support
    and what it depends on.  Never a flat string list."""
    out: list[dict] = []
    for goal in structured.knowledge_goals:
        out.append({
            "goal_id": goal.id,
            "topic": goal.topic,
            "question": goal.question,
            "entities": list(goal.entities),
            "relations": list(goal.relations),
            "dependencies": list(goal.depends_on),
            "required_evidence": list(goal.required_evidence),
            "queries": list(goal.queries),
            "confidence": round(goal.confidence, 3),
            "satisfied": False,
        })
    return out


def response_outline(structured: StructuredRequest) -> list[dict]:
    """The response's own plan: introduction, one section per
    knowledge goal IN ITS DEPENDENCY ORDER, the shaping the request
    asked for, a synthesis and a close.  Sections come from knowledge
    coverage — a response requirement never becomes a section about
    itself."""
    outline: list[dict] = [{"role": "introduction", "heading": ""}]
    for goal in structured.knowledge_goals:
        outline.append({
            "role": "section",
            "heading": _heading_for(goal),
            "goal_id": goal.id,
            "depends_on": list(goal.depends_on),
        })
    if structured.has_requirement("examples"):
        outline.append({"role": "examples", "heading": "Examples"})
    if structured.has_requirement("audience"):
        outline.append({"role": "audience", "heading": ""})
    outline.append({"role": "synthesis", "heading": ""})
    outline.append({"role": "conclusion", "heading": ""})
    return outline


def _heading_for(goal: KnowledgeGoal) -> str:
    """A readable section heading for a goal: the question it asks,
    framed as a heading rather than copied as an answer.  A question
    the request asked in one long clause falls back to the goal's own
    subject — a heading that runs off the line is no heading at all."""
    question = re.sub(r"\s+", " ", goal.question or "").strip()
    topic = re.sub(r"\s+", " ", goal.topic or "").strip()
    if not question:
        return _as_heading(topic)
    if len(question) <= 64:
        if not _WH.match(question):
            question = f"How {question}" if goal.relations else question
        question = question[:1].upper() + question[1:]
        return question.rstrip("?")[:70]
    return _as_heading(topic or question)


def _as_heading(text: str) -> str:
    text = text.strip(" \t,.;:!?-")
    if not text:
        return text
    return text[:1].upper() + text[1:]


def section_heading(goal) -> str:
    """The heading the response gives a knowledge goal: the ask the
    goal states, written as a heading rather than as an answer."""
    return _heading_for(goal)
