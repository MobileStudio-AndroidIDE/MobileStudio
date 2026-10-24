"""Long-form response generation: the OUTPUT GENERATION layer.

A response's length and structure are decided from the REQUEST's own
grammar (depth markers, content acts, enumerations, subject counts)
and the EVIDENCE the turn actually holds — never from a canned
question→length table.  The budget bounds the output; the sections
are planned from the requirements the request states and the material
the retrieval produced; the generation is incremental (per section);
the verification measures coverage / repetition / contradiction /
termination before the text is allowed out.

    Request Grammar → ResponseBudget → Requirements → Sections
    → GenerationState → incremental generation → verification
    → bounded revision
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_DEPTH_MARKERS = (
    "in detail", "in more detail", "more detailed", "in depth",
    "thoroughly", "comprehensively", "step by step", "step-by-step",
    "from start to finish", "end to end", "end-to-end", "in full",
    "the full picture", "every aspect", "covers everything",
    "자세하게", "상세하게", "자세히", "단계별로", "단계별",
    "철저히", "구체적으로", "처음부터 끝까지", "전체적으로",
)

_BRIEF_MARKERS = (
    "briefly", "concisely", "in short", "in a nutshell", "one sentence",
    "short answer", "one line", "tldr",
    "짧게", "간단히", "간단하게", "요약하면", "한 줄로", "한줄로", "간략히",
)

_CONTENT_ACTS = (
    "explain", "describe", "show", "teach", "compare", "contrast",
    "define", "walk me through", "walk through", "illustrate",
    "summarize", "analyse", "analyze", "demonstrate", "enumerate",
    "cover", "discuss", "clarify", "break down", "tell", "list",
    "설명해", "설명해줘", "알려줘", "비교해", "비교해줘", "정의해",
    "가르쳐", "보여줘", "나열해", "목록화",
)

_DISCOURSE_RESET = re.compile(
    r"(?:actually,?\s*)?"
    r"(?:forget(?:\s+the)?\s+(?:last|previous|earlier)\s+"
    r"(?:topic|question|thing|one)"
    r"|never\s*mind(?:\s+that)?"
    r"|ignore\s+(?:that|the\s+last\s+one)"
    r"|지난\s*(?:주제|질문)은?\s*잊어"
    r"|아무튼\s*(?:지난\s*주제는?\s*)?)",
    re.IGNORECASE,
)

_ENUM_PREFIX = re.compile(
    r"^(?:with|including|covering|such\s+as|and|plus|&|"
    r"을\s*포함(?:해서|하여|해)|와\s*함께)\s+",
    re.IGNORECASE)

_EXAMPLE_DEMAND = re.compile(
    r"((?:several|multiple|many|various|practical|real[\s-]world|"
    r"concrete|different|plural)\s+examples?"
    r"|examples?\s+of\s+each"
    r"|예시(?:들|를\s*여러(?:\s*개)?|를\s*다양|를\s*구체)"
    r"|여러\s*예시|실전\s*예시)",
    re.IGNORECASE,
)

_CAUTION_DEMAND = re.compile(
    r"(common\s+mistakes?|pitfalls?|gotchas?|mistakes\s+to\s+avoid"
    r"|things\s+to\s+avoid|what\s+can\s+go\s+wrong"
    r"|자주\s*하는\s*실수|주의할\s*점|함정)",
    re.IGNORECASE,
)

_PROCEDURE_DEMAND = re.compile(
    r"(step\s+by\s+step|the\s+(?:whole\s+)?procedure|how\s+do\s+i\s+"
    r"|how\s+to\s+|instructions?|recipe|단계별|절차|방법)",
    re.IGNORECASE,
)

_AUDIENCE_DEMAND = re.compile(
    r"(for\s+(?:a\s+)?(?:beginner|novice|layperson|non[\s-]expert|"
    r"child|student)s?|beginner[\s-]friendly|입문자|초보자|비전문가)",
    re.IGNORECASE,
)

_CLAUSE_SPLIT = re.compile(
    r"\s*(?:,\s*|\.\s+|;\s+|\n+|\bthen\b\s*|그리고\s+|다음에\s+|"
    r"그\s*다음\s*)",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z+#'`]*|[가-힣]{2,}")

_STOP = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "it", "is",
    "are", "was", "were", "be", "been", "of", "to", "in", "on", "at",
    "for", "from", "with", "by", "as", "and", "or", "but", "if", "then",
    "so", "because", "than", "into", "over", "under", "about", "what",
    "which", "who", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might", "must",
    "please", "tell", "give", "show", "let", "make", "create", "write",
    "explain", "describe", "define", "help", "find", "check", "compare",
    "list", "also", "more", "very", "really", "just", "simply",
    "me", "my", "your", "you", "i", "we", "they", "he", "she",
})

_GAP_MARKERS = (
    "could not stand up", "could not find", "not enough evidence",
    "length budget", "찾지 못", "근거를 찾지 못", "확인되지 않",
    "길이 예산",
)

_TERMINAL = ".!?…\"')]}"


@dataclass
class ResponseBudget:
    """The output's length bounds, decided from the request's grammar
    and the turn's evidence — the generation never wanders past the
    maximum and never pads below the minimum with invented content."""
    minimum: int = 300
    target: int = 900
    maximum: int = 2500
    reason: str = "default"
    signals: list[str] = field(default_factory=list)
    warranted: bool = False

    def warrants_longform(self) -> bool:
        return bool(self.warranted)

    def to_dict(self) -> dict:
        return {
            "minimum": self.minimum,
            "target": self.target,
            "maximum": self.maximum,
            "reason": self.reason,
            "signals": list(self.signals),
            "warranted": self.warranted,
        }

    @classmethod
    def from_spec(cls, spec) -> "ResponseBudget":
        """The bounds a ResponseSpecification states — the generation
        reads the SAME numbers the requirement analysis wrote, so the
        specification and the budget can never disagree."""
        data = dict(spec or {})
        if not data.get("target_length"):
            raise ValueError("specification states no target length")
        minimum = int(data.get("minimum_length") or 0)
        target = int(data.get("target_length") or 0)
        maximum = int(data.get("maximum_length") or 0)
        return cls(
            minimum=minimum or 300,
            target=target or 900,
            maximum=maximum or max(target + 3000, 2500),
            reason=str(data.get("depth") or "specification"),
            signals=list(data.get("constraints") or []),
            warranted=bool(data.get("warrants_longform")),
        )

    @classmethod
    def from_dict(cls, data) -> "ResponseBudget":
        data = dict(data or {})
        return cls(
            minimum=int(data.get("minimum", 300) or 300),
            target=int(data.get("target", 900) or 900),
            maximum=int(data.get("maximum", 2500) or 2500),
            reason=str(data.get("reason", "default") or "default"),
            signals=list(data.get("signals") or []),
            warranted=bool(data.get("warranted")),
        )


@dataclass
class Section:
    """One planned section of a long response: its heading, its
    purpose, the requirement it serves and the evidence sentences it
    may use."""
    heading: str = ""
    purpose: str = "content"
    requirement: dict = field(default_factory=dict)
    material: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "purpose": self.purpose,
            "requirement": dict(self.requirement),
            "material": list(self.material),
        }


@dataclass
class GenerationState:
    """The incremental generation's shared state: what has been
    completed, what evidence has been spent, which terms were chosen,
    and what the response still owes the request."""
    request: str = ""
    semantic_representation: dict = field(default_factory=dict)
    response_plan: dict = field(default_factory=dict)
    completed_sections: list[str] = field(default_factory=list)
    active_section: str = ""
    unresolved_requirements: list[dict] = field(default_factory=list)
    used_evidence: list[str] = field(default_factory=list)
    terminology: dict = field(default_factory=dict)
    generated_artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "semantic_representation": dict(self.semantic_representation),
            "response_plan": dict(self.response_plan),
            "completed_sections": list(self.completed_sections),
            "active_section": self.active_section,
            "unresolved_requirements": [
                dict(r) for r in self.unresolved_requirements],
            "used_evidence": list(self.used_evidence),
            "terminology": dict(self.terminology),
            "generated_artifacts": dict(self.generated_artifacts),
        }


def _words(text: str) -> list[str]:
    """The content words of a fragment: lowercase, stop class dropped."""
    out = []
    for raw in _WORD_RE.findall((text or "").lower()):
        w = raw.strip("'\"`")
        if not w or w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


def _normalize(sentence: str) -> str:
    return re.sub(r"\s+", " ", (sentence or "").strip().lower()).rstrip(" .")


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(
        r"(?<=[.!?。])\s+", (text or "").strip())]
    return [p for p in parts if p]


def _paragraphs(block: str, limit: int = 700) -> list[str]:
    """A section's body in paragraphs of at most `limit` characters,
    cut at sentence boundaries.  One section's material stays several
    segments instead of a single wall, so a length cap trims the TAIL
    of the body and never has to drop a whole section's evidence."""
    sentences = _split_sentences(block)
    paragraphs: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > limit:
            paragraphs.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        paragraphs.append(current)
    return paragraphs


def _has_marker(text: str, markers) -> bool:
    lower = (text or "").lower()
    return any(m in lower for m in markers)


def _content_act_count(text: str) -> int:
    """How many CONTENT acts this request performs: clauses whose
    predicate asks for knowledge work.  A discourse reset ("forget the
    last topic") is removed first — it changes the floor, not the
    subject."""
    stripped = _DISCOURSE_RESET.sub(" ", text or "")
    count = 0
    for clause in _CLAUSE_SPLIT.split(stripped):
        low = clause.lower()
        if any(v in low for v in _CONTENT_ACTS):
            count += 1
    return count


def _enumerated_items(text: str) -> list[str]:
    """The request's own enumeration: the items after an enumeration
    intro ("including A, B, and C") — structural, never semantic."""
    items: list[str] = []
    for m in re.finditer(
            r"(?:including|with|covering|such\s+as)\s+([^.;?]+)",
            (text or ""), re.IGNORECASE):
        segment = m.group(1)
        parts = re.split(r",\s*|\s+and\s+|\s*&\s*|、|과\s+|와\s+", segment)
        for p in parts:
            p = _ENUM_PREFIX.sub("", p.strip(" \t."))
            if len(p) >= 3:
                items.append(p)
    for m in re.finditer(
            r"(?:을\s*포함(?:해서|하여|해)|다음과\s*같이)\s+([^.;?]+)",
            (text or "")):
        parts = re.split(r",\s*|、|을\s+|와\s+|과\s+", m.group(1))
        for p in parts:
            p = _ENUM_PREFIX.sub("", p.strip())
            if len(p) >= 2:
                items.append(p)
    return items


def _subject_list(target: str) -> list[str]:
    """The comparison's subject set: a target naming three or more
    members ("boiling, steaming, frying, and roasting")."""
    if not target:
        return []
    parts = re.split(r",\s*|\s+and\s+|\s*·\s*|、|과\s+|와\s+",
                     target, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        cleaned = part.strip(" \t.")
        cleaned = re.sub(r"^(?:and|or|&)\s+", "", cleaned,
                         flags=re.IGNORECASE)
        if len(cleaned) >= 2:
            out.append(cleaned)
    return out


# The WH-coordinated ask: a clause the request itself states as a
# separate thing to tell ("how stars are born, how they change over
# time").  The frame a clause opens with is grammar; the ask after it
# is the part.
_WH_ASK = re.compile(
    r"^\s*(?:please\s+|also\s+|and\s+|then\s+|"
    r"explain\s+|describe\s+|tell\s+me\s+(?:about\s+)?|cover\s+|"
    r"list\s+|compare\s+)?"
    r"(?:what|why|when|where|who|how|which)\b", re.IGNORECASE)

# The span through a subject: a start, a path, a horizon.  The turn
# asks its subject to be covered OVER a course, so one stopping point
# cannot answer it.
_PROGRESSION_FRAME = re.compile(
    r"\b(?:from\s+(?:the\s+)?(?:beginning|scratch|start|zero|end)|"
    r"(?:over|across|during)\s+(?:the\s+|a\s+)?(?:next|coming|"
    r"following|course\s+of)\s+[\w\s]{1,24}|"
    r"step\s+by\s+step|from\s+start\s+to\s+end|"
    r"처음부터|시작부터|단계별로|차근차근)", re.IGNORECASE)


# The FRAME a part opens with: the ask's own performative ("Explain",
# "Teach me", "I want to learn how to") is how the request is
# addressed — the part itself is what stands after it, and it is what
# the response's section is called.
_PART_FRAME = re.compile(
    r"^(?:please\s+|just\s+|also\s+|could\s+you\s+|can\s+you\s+|"
    r"would\s+you\s+|explain\s+|describe\s+|teach\s+(?:me\s+)?|"
    r"show\s+(?:me\s+)?|tell\s+me\s+|cover\s+|list\s+|discuss\s+|"
    r"define\s+|summar\w*\s+|i\s+want\s+to\s+(?:learn|know|"
    r"understand)(?:\s+how\s+to)?\s+)",
    re.IGNORECASE)


def _without_frame(phrase: str) -> str:
    return _PART_FRAME.sub("", str(phrase or ""), count=1).strip()


def _opens_with_act(clause: str) -> bool:
    """Whether a clause OPENS with a content act: the verb that names
    what the turn asks to be told.  Grammar — the clause's own mood."""
    head = str(clause or "").strip().lower()
    for marker in _CONTENT_ACTS:
        token = str(marker).strip().lower()
        if head.startswith(token):
            return True
    return False


_COORDINATION = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)

# a finite predicate at the segment's own head: a subject and the verb
# that predicates of it (an auxiliary, a modal, or a do-form)
_FINITE_PREDICATE = re.compile(
    r"^\s*(?:[A-Za-z][\w'\-]*\s+)?(?:is|are|was|were|am|do|does|did|"
    r"has|have|had|can|could|will|would|should|may|might|must|"
    r"shows?|makes?|gives?|keeps?|needs?|takes?|puts?|runs?|works?)"
    r"(?:\s+\w+)", re.IGNORECASE)


def _is_clausal(segment: str) -> bool:
    """Whether a segment STATES something: it opens with a WH-ask,
    opens with a content act of its own, or carries a finite predicate
    at its head.  Whether a coordination joins CLAUSES (more than one
    ask) or NOMINALS (one ask naming several things) is decided from
    that grammar — never from what the segments happen to say."""
    text = str(segment or "").strip()
    if len(text) < 5:
        return False
    if _WH_ASK.match(text) or _opens_with_act(text):
        return True
    return bool(_FINITE_PREDICATE.match(text))


def _split_coordinated_clauses(clause: str) -> list[str]:
    """A coordination of CLAUSES states more than one ask; a
    coordination of nouns inside one act ("tell me about A and B")
    states one.  The conjunction is read clause-wise: both sides must
    be clauses before the join counts as a separate ask."""
    text = str(clause or "")
    if not text.strip():
        return []
    for match in _COORDINATION.finditer(text):
        left, right = text[:match.start()], text[match.end():]
        if _is_clausal(left) and _is_clausal(right):
            return (_split_coordinated_clauses(left)
                    + _split_coordinated_clauses(right))
    return [text]


def request_aspects(request: str) -> list[str]:
    """The PARTS this request itself asks for, read from its own
    structure: its enumerated items, the poles of its scope, the
    coordinated asks it states one after another.

    Each part is something the response owes — a section and its own
    evidence requirement.  The request's own words, never a topic
    keyword and never a sentence copied as an answer."""
    text = str(request or "").strip()
    if not text:
        return []
    out: list[str] = []

    def add(phrase: str) -> None:
        phrase = re.sub(r"\s+", " ", str(phrase or "")).strip(" \t,.;:!?-")
        phrase = _without_frame(phrase)
        if len(phrase) < 4:
            return
        phrase = phrase[:140]
        if any(phrase.lower() == e.lower() for e in out):
            return
        out.append(phrase)

    for item in _enumerated_items(text):
        add(item)
    # a sentence is a predication: the parts are read inside it, so a
    # question mark opens the ask that follows it
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        clauses = [part
                   for chunk in _CLAUSE_SPLIT.split(sentence)
                   for part in _split_coordinated_clauses(chunk)
                   if str(part or "").strip()]
        index = 0
        while index < len(clauses):
            head = str(clauses[index]).strip()
            if _WH_ASK.match(head):
                add(head)
                index += 1
            elif _opens_with_act(head):
                # the act and its object list are ONE ask: the list is
                # what is to be told, not a separate turn
                group = [head]
                index += 1
                while index < len(clauses):
                    following = str(clauses[index]).strip()
                    if _WH_ASK.match(following) or _opens_with_act(following):
                        break
                    group.append(following)
                    index += 1
                add(", ".join(group))
            else:
                index += 1
    for m in re.finditer(r"\bfrom\s+(.{3,60}?)\s+to\s+(.{3,60}?)(?=[,.;]|$)",
                         text, re.IGNORECASE):
        add(m.group(1))
        add(m.group(2))
    return out[:6]


def _stem(word: str) -> str:
    """The comparison form of a word for coverage: its first five
    letters, so `languages` and `language` are one word for the
    purpose of asking whether a part was SAID."""
    return str(word or "").lower()[:5]


def account_for_parts(text: str, request: str, plan=None,
                      language: str = "en") -> str:
    """A SHORT answer to a MULTI-part ask accounts for every part it
    owes: a part the wording never reaches is STATED as not covered —
    never dropped silently and never filled with invented content.

    A single-part ask keeps its natural length: nothing is padded to
    look complete, and a part the evidence did stand up is left alone."""
    body = str(text or "")
    parts = request_aspects(request)
    if len(parts) < 2:
        return body
    spoken = {_stem(w) for w in _words(body)}
    retrieval = getattr(plan, "retrieval", None) or {}
    has_evidence = bool(getattr(plan, "evidence", None)
                        or retrieval.get("evidence"))
    korean = str(language or "").lower().startswith("ko")
    trailing: list[str] = []
    for part in parts:
        terms = [w for w in _words(part) if w.lower() not in _STOP]
        if not terms:
            continue
        if any(_stem(term) in spoken for term in terms):
            continue
        if korean:
            trailing.append(
                f"이 답변에서 다루지 못한 부분이 있습니다: {part}."
                if has_evidence else
                f"이 부분은 뒷받침할 서지를 세울 수 없었습니다: {part}.")
        else:
            trailing.append(
                f"One part of the request is not covered here: {part}."
                if has_evidence else
                f"I could not stand up evidence for one part: {part}.")
    if not trailing:
        return body
    return (body.rstrip() + " " + " ".join(trailing)).strip()


def progression_frame(request: str) -> bool:
    """Whether the request frames a COURSE through its subject: a
    start, a span, a horizon.  A turn that asks for a path asks for
    more than one stopping point."""
    return bool(_PROGRESSION_FRAME.search(str(request or "")))


def compute_response_budget(request: str, plan, semantic=None) -> ResponseBudget:
    """Decide the response's length bounds from the request's own
    grammar: explicit depth markers, content-act count, enumerations,
    subject multiplicity, length, and the register the turn asked for.

    A brief register caps the response short; a simple register never
    warrants long-form; a request with no structural depth signal keeps
    the default budget.  No question is looked up — the DECISION is a
    property of the request's structure."""
    text = str(request or "")
    intent = str(getattr(plan, "intent", "") or "")
    style = str(getattr(plan, "style", "") or "")
    content_intents = {
        "information_query", "explanation_request",
        "procedural_instruction",
        # an investigation asks the same of its sources: the depth the
        # request states sets the bounds the evidence must fill
        "research",
    }
    if intent not in content_intents or style == "simple":
        return ResponseBudget(reason="register")
    signals: list[str] = []
    if _has_marker(text, _BRIEF_MARKERS):
        return ResponseBudget(
            minimum=150, target=400, maximum=900,
            reason="brief register", signals=["brief marker"],
            warranted=False)
    if _has_marker(text, _DEPTH_MARKERS):
        signals.append("depth markers")
    acts = _content_act_count(text)
    if acts >= 2:
        signals.append(f"{acts} content acts")
    items = _enumerated_items(text)
    if len(items) >= 3:
        signals.append(f"{len(items)} enumerated requirements")
    parts = request_aspects(text)
    if len(parts) >= 2:
        signals.append(f"{len(parts)} requested parts")
    if progression_frame(text):
        signals.append("progression frame")
    target = str(getattr(plan, "target", "") or "")
    subjects = _subject_list(target)
    if len(subjects) >= 3:
        signals.append(f"{len(subjects)} subjects")
    words = len(_WORD_RE.findall(text))
    if words >= 35:
        signals.append(f"{words} request words")
    example = _EXAMPLE_DEMAND.search(text)
    if example and re.search(r"(?:several|multiple|many|various|여러|여러\s*개|다양)",
                             example.group(1), re.IGNORECASE):
        signals.append("multi-example demand")
    caution = _CAUTION_DEMAND.search(text)
    if caution:
        signals.append("caution demand")
    if not signals:
        return ResponseBudget(reason="default")
    aspects = max(len(items), len(subjects), len(parts))
    base = 3000
    target_len = base + 500 * max(0, aspects - 2) + 500 * max(0, acts - 1)
    target_len = min(target_len, 10000)
    return ResponseBudget(
        minimum=min(2000, target_len),
        target=target_len,
        maximum=min(10000, target_len + 3000),
        reason="structural depth",
        signals=signals,
        warranted=True,
    )


def build_response_specification(request: str, plan,
                                 budget: ResponseBudget) -> dict:
    """What THIS response must be, in one structure: the depth the
    request asked for, its bounds, the parts it owes and the constraints
    it set — read from the request's own grammar, never guessed from a
    template."""
    text = str(request or "")
    depth = "deep" if budget.warrants_longform() else (
        "brief" if budget.reason == "brief register" else "normal")
    example = _EXAMPLE_DEMAND.search(text)
    example_count = 0
    if example:
        phrase = str(example.group(1) or "").lower()
        example_count = (3 if re.search(
            r"(?:several|multiple|many|various|여러|다양)", phrase,
            re.IGNORECASE) else 1)
    audience = _AUDIENCE_DEMAND.search(text)
    caution = _CAUTION_DEMAND.search(text)
    constraints: list[str] = []
    if caution:
        constraints.append(str(caution.group(1) or "").strip())
    if re.search(r"\b(?:briefly|concisely|in short|shortly)\b", text,
                 re.IGNORECASE):
        constraints.append("state it briefly")
    try:
        sections = len(extract_requirements(request, plan, budget))
    except Exception:
        sections = 0
    return {
        "depth": depth,
        "detail_level": str(getattr(plan, "style", "") or ""),
        "target_length": int(budget.target),
        "minimum_length": int(budget.minimum),
        "maximum_length": int(budget.maximum),
        "section_count": max(0, sections),
        "example_count": example_count,
        "explanation_style": str(getattr(plan, "style", "") or ""),
        "complexity": "",
        "audience": str(audience.group(1) or "").strip() if audience
        else "",
        "constraints": constraints,
        "warrants_longform": bool(budget.warrants_longform()),
    }


def extract_requirements(request: str, plan, budget: ResponseBudget) -> list[dict]:
    """What the response OWES this request, extracted from its own
    structure: enumerated items, per-subject asks, the example / caution
    / procedure / audience demands it states, and each content act."""
    text = str(request or "")
    target = str(getattr(plan, "target", "") or "")
    requirements: list[dict] = []
    seen: set[str] = set()

    def add(kind: str, phrase: str) -> None:
        phrase = re.sub(r"\s+", " ", phrase).strip(" \t,.;:!?")
        if kind in ("aspect", "act"):
            phrase = _without_frame(phrase)
        if len(phrase) < 2:
            return
        key = phrase.lower()
        if key in seen:
            return
        seen.add(key)
        requirements.append({"kind": kind, "phrase": phrase,
                             "terms": _words(phrase)})

    for item in _enumerated_items(text):
        if _EXAMPLE_DEMAND.fullmatch(item) or \
                _CAUTION_DEMAND.fullmatch(item) or \
                _AUDIENCE_DEMAND.fullmatch(item):
            continue
        add("enumeration", item)
    for aspect in request_aspects(text):
        if _EXAMPLE_DEMAND.fullmatch(aspect) or \
                _CAUTION_DEMAND.fullmatch(aspect) or \
                _AUDIENCE_DEMAND.fullmatch(aspect):
            continue
        add("aspect", aspect)
    subjects = _subject_list(target)
    if len(subjects) >= 2:
        for s in subjects:
            add("subject", s)
    m = _EXAMPLE_DEMAND.search(text)
    if m:
        add("examples", m.group(1))
    m = _CAUTION_DEMAND.search(text)
    if m:
        add("caution", m.group(1))
    m = _AUDIENCE_DEMAND.search(text)
    if m:
        add("audience", m.group(1))
    stripped = _DISCOURSE_RESET.sub(" ", text)
    for clause in _CLAUSE_SPLIT.split(stripped):
        low = clause.lower()
        if any(v in low for v in _CONTENT_ACTS):
            head = clause.strip(" \t,.;:")
            if len(head) >= 8:
                add("act", head)
    if not requirements and _has_marker(text, _DEPTH_MARKERS):
        m2 = re.search(r"(in (?:more )?detail|in depth|thoroughly|"
                       r"comprehensively|step by step|자세하게|상세하게|"
                       r"단계별로)", text, re.IGNORECASE)
        if m2:
            add("depth", m2.group(1))
    return requirements


def _overlap(terms_a: list[str], terms_b: list[str]) -> int:
    if not terms_a or not terms_b:
        return 0
    return len(set(terms_a) & set(terms_b))


def plan_sections(budget: ResponseBudget, requirements: list[dict],
                  plan, material: list[str],
                  language: str = "en") -> list[Section]:
    """Plan the long response's sections from the requirements the
    request states and the material the turn holds.  A requirement with
    no supporting material becomes an honest-gap section — it is never
    filled with invented content."""
    target = str(getattr(plan, "target", "") or "")
    target_terms = _words(target)
    sections: list[Section] = [
        Section(heading="_opening", purpose="framing"),
    ]
    material = [m for m in dict.fromkeys(m for m in material if m)]
    remaining = list(material)
    subject_sections: list[Section] = []
    subjects = _subject_list(target)
    if len(subjects) >= 2:
        for s in subjects:
            st = set(_words(s))
            picked = [m for m in remaining
                      if st & set(_words(m))]
            if picked:
                remaining = [m for m in remaining if m not in picked]
            subject_sections.append(
                Section(heading=s, purpose="core",
                        requirement={"kind": "subject", "phrase": s,
                                     "terms": _words(s)},
                        material=picked))
    sections.extend(subject_sections)
    for req in requirements:
        if any(r.get("phrase", "").lower() == req["phrase"].lower()
               for s in subject_sections
               for r in [s.requirement]):
            continue
        terms = set(req["terms"]) - set(target_terms)
        if not terms:
            # The requirement IS the target ("about programming
            # languages" asked for what the turn is about): subtracting
            # the target would leave nothing to match, so material about
            # the target fills it — a section about the subject the
            # request named, never a gap for want of terms.
            terms = set(req["terms"])
        if req.get("kind") == "examples":
            terms = {t for t in terms
                     if t in ("example", "instance", "case", "예시", "사례")}
        elif req.get("kind") == "caution":
            terms = {t for t in terms
                     if t in ("mistake", "pitfall", "gotcha", "error",
                              "실수", "주의", "함정")}
        picked: list[str] = []
        if terms:
            threshold = max(1, -(-len(terms) // 2))
            picked = [s for s in remaining
                      if len(terms & set(_words(s))) >= threshold]
            if picked:
                remaining = [m for m in remaining if m not in picked]
        if not picked and req.get("kind") == "act":
            continue
        purpose = {
            "examples": "examples", "caution": "caution",
            "audience": "core", "subject": "core",
        }.get(req["kind"], "core")
        sections.append(Section(
            heading=req["phrase"], purpose=purpose,
            requirement=req, material=picked))
    if not requirements and not subject_sections and remaining:
        buckets: dict[str, list[str]] = {}
        order: list[str] = []
        for sentence in remaining:
            keys = [w for w in _words(sentence)
                    if w not in target_terms and len(w) > 3]
            key = keys[0] if keys else "_core"
            if key not in buckets:
                buckets[key] = []
                order.append(key)
            buckets[key].append(sentence)
        for key in order[:6]:
            sections.append(Section(
                heading=key.capitalize(), purpose="core",
                material=buckets[key]))
        leftover = [s for k in order[6:] for s in buckets[k]]
        if leftover:
            sections.append(Section(
                heading="Further details", purpose="core",
                material=leftover))
    elif remaining and requirements:
        sections.append(Section(
            heading="Further details" if not
            str(language or "").startswith("ko") else "세부 내용",
            purpose="core",
            material=remaining))
    sections.append(Section(heading="_summary", purpose="summary"))
    if getattr(plan, "evidence", None):
        sections.append(
            Section(heading="_sources", purpose="attribution"))
    return sections


def _framing_sentence(heading: str, purpose: str, plan, state,
                      language: str) -> str:
    """The long response's own opening and close: NONE are written.

    Sentences that announce the REQUEST or the RESPONSE PROCESS
    ("What follows about X …", "To pull it together …", "'X'의 요청
    범위를 …") describe the processing, not the subject; the body and
    the attribution block carry everything the response owes."""
    return ""


def _gap_sentence(heading: str, language: str) -> str:
    if str(language or "").startswith("ko"):
        return (f"'{heading}'에 해당하는 근거를 찾지 못해, "
                f"이 항목은 비워 두었습니다.")
    return (f"I could not stand up evidence for '{heading}', so this "
            f"section is left open.")


def _sources_block(plan, language: str) -> list[str]:
    evidence = list(getattr(plan, "evidence", None) or [])
    if not evidence:
        return []
    ko = str(language or "").startswith("ko")
    lines: list[str] = []
    seen: set[str] = set()
    for src in evidence[:8]:
        key = str(src or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        lines.append(f"{'출처' if ko else 'Source'}: {key}")
    return ["\n".join(lines)] if lines else []


def generate_long_form(budget: ResponseBudget, sections: list[Section],
                       plan, state: GenerationState,
                       language: str = "en") -> tuple:
    """Assemble the long response section by section: material
    sentences in order (each spent once), framing where the evidence
    is silent, an honest gap where it is missing, and a closing that
    restates the request's own terms."""
    language = str(language or getattr(plan, "requested_language", "")
                   or "en")
    target = str(getattr(plan, "target", "") or "")
    state.terminology.setdefault("__target__", target)
    used: set[str] = {_normalize(u) for u in state.used_evidence}
    unresolved = [dict(r) for r in state.unresolved_requirements] or []
    planned = [s.heading for s in sections
               if s.purpose not in ("framing",)]
    state.generated_artifacts["planned_sections"] = planned
    state.generated_artifacts["requirements"] = [
        dict(r) for r in unresolved]
    state.generated_artifacts.setdefault("material_requirements", [])
    state.generated_artifacts.setdefault("deferred_requirements", [])
    chunks: list[str] = []
    body_so_far: list[str] = []
    core_count = len([s for s in sections if s.purpose == "core"])
    request_norm = " ".join(str(getattr(state, "request", "") or "")
                            .lower().split()).rstrip("?!.")
    acc = 0
    for section in sections:
        if section.purpose in ("summary", "attribution", "framing"):
            continue
        if acc >= budget.target and not section.requirement:
            try:
                state.generated_artifacts["planned_sections"].remove(
                    section.heading)
            except ValueError:
                pass
            continue
        body: list[str] = []
        for sentence in section.material:
            key = _normalize(sentence)
            if not key or key in used:
                continue
            used.add(key)
            body.append(sentence)
        # The section's material is COMPOSED, never concatenated (the
        # same rule as the document form): cleaned, meta dropped,
        # near-duplicates removed, ranked and condensed.
        from brain.evidence_compose import compose_body
        composed = compose_body(
            [{"text": s, "source": ""} for s in body],
            question=str(getattr(state, "request", "") or ""),
            language="ko" if str(language or "").startswith("ko")
            else "en",
            max_chars=max(1800, int(getattr(budget, "maximum", 2500)
                                     or 2500)),
            max_words=80)
        cleaned = composed.get("body") or ""
        body = [cleaned] if cleaned else []
        if body:
            for item in (composed.get("kept") or []):
                value = item.get("text") if isinstance(item, dict) else ""
                if value:
                    state.used_evidence.append(value)
            block = " ".join(body)
            heading_norm = " ".join(str(section.heading or "")
                                    .lower().split()).rstrip("?!.")
            show_heading = (core_count >= 2
                            and section.purpose == "core"
                            and heading_norm
                            and heading_norm != request_norm)
            for index, paragraph in enumerate(_paragraphs(block) or [block]):
                chunks.append(f"{section.heading}\n{paragraph}"
                              if index == 0 and show_heading
                              else paragraph)
            acc += len(block) + (len(section.heading) + 1
                                 if show_heading else 0) + 4
            body_so_far.extend(body)
            state.completed_sections.append(section.heading)
            if section.requirement:
                unresolved = [
                    r for r in unresolved
                    if r.get("phrase", "").lower()
                    != section.requirement.get("phrase", "").lower()]
                state.generated_artifacts[
                    "material_requirements"].append(
                        section.requirement.get("phrase", ""))
            state.active_section = section.heading
            continue
        if section.requirement:
            pw = set(section.requirement.get("terms") or []) or set(
                _words(section.requirement.get("phrase", "")))
            if any(pw & set(_words(s)) and not _is_gap_sentence(s)
                   for s in body_so_far):
                state.generated_artifacts["material_requirements"].append(
                    section.requirement.get("phrase", ""))
                state.completed_sections.append(section.heading)
                state.active_section = section.heading
                unresolved = [
                    r for r in unresolved
                    if r.get("phrase", "").lower()
                    != section.requirement.get("phrase", "").lower()]
                continue
            gap = _gap_sentence(section.heading, language)
            chunks.append(f"{section.heading}\n{gap}")
            acc += len(gap) + len(section.heading) + 4
            state.completed_sections.append(section.heading)
            state.active_section = section.heading
            continue
        framing = _framing_sentence(
            section.heading, section.purpose, plan, state, language)
        if framing:
            chunks.append(framing)
            acc += len(framing) + 2
            state.completed_sections.append(section.heading)
    all_reqs = list(
        state.generated_artifacts.get("requirements") or [])
    material_covered = set(
        state.generated_artifacts.get("material_requirements") or [])
    body_sentences = _split_sentences("\n".join(chunks))
    open_reqs: list[dict] = []
    for req in all_reqs:
        if req.get("phrase", "") in material_covered:
            continue
        pw = set(req.get("terms") or []) or set(
            _words(req.get("phrase", "")))
        if any(pw & set(_words(s)) and not _is_gap_sentence(s)
               for s in body_sentences):
            continue
        open_reqs.append(req)
    state.unresolved_requirements = open_reqs
    parts: list[str] = []
    if state.used_evidence:
        opening = _framing_sentence(
            "", "framing", plan, state, language)
        if opening:
            parts.append(opening)
        parts.extend(chunks)
        summary = _framing_sentence(
            "", "summary", plan, state, language)
        if summary:
            parts.append(summary)
    elif chunks:
        parts.extend(chunks)
    else:
        target_name = target or "this request"
        if str(language or "").startswith("ko"):
            parts.append(f"'{target_name}'에 대한 근거를 찾지 못해, "
                         f"장문으로 쓸 근거가 아직 없습니다.")
        else:
            parts.append(f"I could not stand up enough evidence for "
                         f"'{target_name}' to write this up in "
                         f"depth.")
    parts.extend(_sources_block(plan, language))
    text = "\n\n".join(parts) if parts else ""
    if text and text[-1] not in _TERMINAL:
        text += "."
    text = _enforce_maximum(text, budget, sections, plan, state, language)
    report = verify_long_form(text, state, budget, plan)
    return text, state, report


def _enforce_maximum(text: str, budget: ResponseBudget,
                     sections: list[Section], plan, state,
                     language: str) -> str:
    """The hard cap: when the sections together overrun the budget's
    maximum, trailing content segments are dropped (their requirements
    stay recorded — the verification notices and the revision marks
    them honestly), while the closing and the sources are kept."""
    if len(text) <= budget.maximum:
        return text
    segments = [s for s in text.split("\n\n") if s.strip()]
    tail: list[str] = []
    content: list[str] = []
    for segment in segments:
        head = segment.split("\n", 1)[0].strip()
        low = segment.lower()
        if low.startswith("to pull it together") or \
                head in ("_summary",) or \
                segment.startswith("정리하면") or \
                segment.startswith(("Source:", "출처:")):
            tail.append(segment)
        else:
            content.append(segment)
    while len(content) > 1 and \
            len("\n\n".join(content + tail)) > budget.maximum:
        dropped = content.pop()
        head = dropped.split("\n", 1)[0].strip()
        for req in state.unresolved_requirements:
            if req.get("phrase", "") == head:
                state.generated_artifacts.setdefault(
                    "deferred_requirements", []).append(head)
    out = "\n\n".join(content + tail)
    if out and out[-1] not in _TERMINAL:
        out += "."
    return out


def _predicate_subject(sentence: str, token: str) -> str:
    """The word a polarity expression is PREDICATED of: the last
    content word before it ("solar power increases" → power)."""
    low = f" {sentence.lower()} "
    index = low.find(token)
    if index < 0:
        return ""
    words = low[:index].split()
    for word in reversed(words):
        if word not in _STOP:
            return word
    return words[-1] if words else ""


def _polarity_conflict(a: str, b: str) -> bool:
    """Two sentences CONFLICT when they predicate OPPOSITES of the
    SAME subject: the polarity expression must be predicated of a word
    both sentences name.  Two facts about two subjects ("a programming
    language is an engineered language" beside "Y is not considered a
    programming language") are not a contradiction — they are two
    facts, and the response may state both."""
    pairs = (
        (" increases ", " decreases "), (" increases ", " reduces "),
        (" possible ", " impossible "), (" can ", " cannot "),
        (" is ", " is not "), (" are ", " are not "),
        (" always ", " never "), (" faster ", " slower "),
        (" higher ", " lower "), (" increases ", " drops "),
    )
    low_a, low_b = f" {a.lower()} ", f" {b.lower()}"
    for x, y in pairs:
        for first, second in ((x, y), (y, x)):
            if first not in low_a or second not in low_b:
                continue
            subject_a = _predicate_subject(a, first)
            subject_b = _predicate_subject(b, second)
            if subject_a and subject_a == subject_b:
                return True
    return False


def _is_gap_sentence(sentence: str) -> bool:
    low = sentence.lower()
    return any(m in low for m in _GAP_MARKERS)


def verify_long_form(text: str, state: GenerationState,
                     budget: ResponseBudget, plan) -> dict:
    """Measure the generated response against the requirements, the
    sections and the budget: coverage (answered or honestly marked
    open), repetition, contradiction, termination, length.  The metrics
    are the checklist; `passed` is their conjunction."""
    headings = set(
        state.generated_artifacts.get("planned_sections") or [])
    body_lines = [line for line in (text or "").splitlines()
                  if line.strip() not in headings]
    sentences = _split_sentences("\n".join(body_lines))
    all_requirements = list(
        state.generated_artifacts.get("requirements")
        or state.unresolved_requirements)
    material_covered = set(
        state.generated_artifacts.get("material_requirements") or [])
    covered = 0
    honest_gaps = 0
    for req in all_requirements:
        phrase = req.get("phrase", "")
        phrase_words = set(req.get("terms") or _words(phrase)) \
            or set(_words(phrase))
        direct = phrase.lower() in {p.lower() for p in material_covered}
        gap = False
        for sentence in sentences:
            sw = set(_words(sentence))
            if not (phrase_words & sw):
                continue
            if _is_gap_sentence(sentence):
                gap = True
            else:
                direct = True
        if direct:
            covered += 1
        elif gap:
            covered += 1
            honest_gaps += 1
    total_req = len(all_requirements)
    coverage = (covered / total_req) if total_req else 1.0
    normalized = [_normalize(s) for s in sentences if _normalize(s)]
    unique = len(set(normalized))
    repetition = (1.0 - unique / len(normalized)) if normalized else 0.0
    contradictions = 0
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            wa, wb = set(_words(sentences[i])), set(_words(sentences[j]))
            if len(wa & wb) >= 3 and _polarity_conflict(
                    sentences[i], sentences[j]):
                contradictions += 1
    stripped = (text or "").strip()
    termination = bool(stripped) and stripped[-1] in _TERMINAL
    planned = list(
        state.generated_artifacts.get("planned_sections") or [])
    deferred = set(
        state.generated_artifacts.get("deferred_requirements") or [])
    planned = [h for h in planned if h not in deferred]
    completed = set(state.completed_sections)
    section_completeness = 1.0
    if planned:
        done = sum(1 for h in planned if h in completed or
                   h in ("_opening", "_summary", "_sources"))
        section_completeness = done / len(planned)
    length = len(text or "")
    shortfall = length < budget.minimum
    issues: list[str] = []
    if coverage < 1.0:
        issues.append("requirement coverage below 1.0")
    if repetition > 0.08:
        issues.append("repetition above threshold")
    if contradictions:
        issues.append("possible contradiction detected")
    if not termination:
        issues.append("response does not terminate")
    if shortfall:
        issues.append("length below the budget minimum")
    if section_completeness < 0.9:
        issues.append("planned sections incomplete")
    passed = not issues
    status = "passed" if passed else (
        "shortfall" if shortfall and len(issues) == 1 else "failed")
    return {
        "passed": passed,
        "status": status,
        "metrics": {
            "length": length,
            "target": budget.target,
            "target_ratio": round(length / max(1, budget.target), 3),
            "minimum": budget.minimum,
            "requirement_coverage": round(coverage, 3),
            "requirements_total": total_req,
            "honest_gaps": honest_gaps,
            "sentences": len(sentences),
            "repetition_ratio": round(repetition, 3),
            "contradictions": contradictions,
            "termination": termination,
            "section_completeness": round(section_completeness, 3),
        },
        "issues": issues,
    }


def revise_long_form(text: str, state: GenerationState,
                     budget: ResponseBudget, plan,
                     report: dict, language: str = "en") -> tuple:
    """The bounded revision: one pass.  Missing requirements are
    marked honestly open (never invented), duplicated sentences are
    dropped, then the response is verified again."""
    language = str(language or getattr(plan, "requested_language", "")
                   or "en")
    sentences = _split_sentences(text)
    seen: set[str] = set()
    deduped: list[str] = []
    for sentence in sentences:
        key = _normalize(sentence)
        if key and key in seen and not _is_gap_sentence(sentence):
            continue
        if key:
            seen.add(key)
        deduped.append(sentence)
    deferred = set(
        state.generated_artifacts.get("deferred_requirements") or [])
    missing = []
    for req in state.unresolved_requirements:
        phrase_words = set(_words(req.get("phrase", "")))
        already = any(
            phrase_words & set(_words(s)) and _is_gap_sentence(s)
            for s in deduped)
        if not already:
            missing.append(req)
    if missing:
        notes: list[str] = []
        for req in missing:
            phrase = req.get("phrase", "")
            if phrase in deferred:
                if str(language).startswith("ko"):
                    notes.append(
                        f"'{phrase}' 항목은 이 응답의 길이 예산 안에서 "
                        f"여기서 멈췄습니다.")
                else:
                    notes.append(
                        f"I stopped before '{phrase}' to stay within "
                        f"this response's length budget.")
            else:
                notes.append(_gap_sentence(phrase, language))
        insert_at = len(deduped)
        for index, sentence in enumerate(deduped):
            head = sentence.strip()
            if head.startswith(("To pull it together", "정리하면")) or \
                    head.startswith(("Source", "출처")):
                insert_at = index
                break
        deduped[insert_at:insert_at] = notes
        state.completed_sections.extend(
            r["phrase"] for r in missing)
        noted = {r.get("phrase", "") for r in missing}
        state.unresolved_requirements = [
            r for r in state.unresolved_requirements
            if r.get("phrase", "") not in noted]
    rebuilt = " ".join(s.strip() for s in deduped if s.strip())
    if rebuilt and rebuilt[-1] not in _TERMINAL:
        rebuilt += "."
    report = verify_long_form(rebuilt, state, budget, plan)
    if report["status"] == "failed" and \
            report["metrics"]["length"] < budget.minimum:
        report["status"] = "shortfall"
    return rebuilt, state, report


# The document path's own markers: the shape the request asked for is
# stated as structure, never as a complaint about evidence.
_EXAMPLE_MARKERS = (
    "for example", "for instance", "such as", "e.g.", "an example",
    "examples include", "one example", "예를 들어", "예시로", "예를 들면",
)


def document_budget(structured, budget: ResponseBudget) -> ResponseBudget:
    """The document's own length bounds.

    A request that asked for depth is answered at the depth it asked
    for — `detailed` and `comprehensive` are stated in the request's
    own grammar, not guessed at from its word count — and a request
    spanning several knowledge goals needs room for each of them.  A
    request that stated neither keeps the bounds its register already
    set: a short turn stays short."""
    goals = len(getattr(structured, "knowledge_goals", []) or [])
    depth = ""
    for req in getattr(structured, "response_requirements", None) or []:
        if getattr(req, "kind", "") == "depth":
            depth = str(getattr(req, "value", "") or "").strip().lower()
    floors = {
        "detailed": (3000, 4200, 7000),
        "comprehensive": (3000, 5200, 8000),
        "in detail": (3000, 4200, 7000),
        "in depth": (3000, 4200, 7000),
        "thoroughly": (3000, 4200, 7000),
        "extensively": (3000, 4200, 7000),
        "full": (3000, 4200, 7000),
    }
    minimum, target, maximum = budget.minimum, budget.target, budget.maximum
    if depth in floors:
        minimum = max(minimum, floors[depth][0])
        target = max(target, floors[depth][1])
        maximum = max(maximum, floors[depth][2])
    elif goals >= 3:
        target = max(target, 400 + 600 * goals)
        minimum = max(minimum, min(target - 900, 1800))
        maximum = max(maximum, target + 2500)
    else:
        return budget
    if (minimum, target, maximum) == (budget.minimum, budget.target,
                                       budget.maximum):
        return budget
    return ResponseBudget(
        minimum=minimum, target=target, maximum=maximum,
        reason=f"{budget.reason}+document",
        signals=list(budget.signals) + [f"{goals} knowledge goals"],
        warranted=True,
    )


def plan_document_sections(budget: ResponseBudget, structured,
                           evidence, plan,
                           language: str = "en") -> list[Section]:
    """The document's response outline as sections: one per knowledge
    goal IN ITS DEPENDENCY ORDER, each holding the sentences the
    retrieval validated for it.

    A goal reads its own source section first, then the rest of the
    validated pool, so one part of the response stays one account.
    The plan is kept whole — what a goal may not reach is recorded as
    deferred rather than written as an apology."""
    from brain.document_semantics import section_heading

    goals = list(getattr(structured, "knowledge_goals", []) or [])
    pool = list(getattr(evidence, "material", []) or [])
    ranked_by_goal = dict(getattr(evidence, "by_goal", {}) or {})
    section_of = list(getattr(evidence, "section_of", []) or [])
    section_order = dict(getattr(evidence, "order_by_goal", {}) or {})

    sections: list[Section] = [Section(heading="_opening",
                                       purpose="framing")]
    reserve = 500 if len(goals) > 1 else 250
    room = max(900, min(budget.target, budget.maximum - 700) - reserve)
    share = max(450, room // max(1, len(goals)))
    taken: set[int] = set()
    leftovers: dict[str, list[int]] = {}
    written_chars = 0

    for goal in goals:
        eligible = [i for i in (ranked_by_goal.get(goal.id) or [])
                    if 0 <= i < len(pool) and i not in taken]
        if len(eligible) < 3:
            want = set(_words(goal.topic)) | set(_words(goal.question))
            eligible += [i for i, sentence in enumerate(pool)
                         if i not in taken and i not in eligible
                         and want & set(_words(sentence))]
        runs: dict[int, list[int]] = {}
        for index in eligible:
            position = section_of[index] if index < len(section_of) else -1
            runs.setdefault(position, []).append(index)
        for run in runs.values():
            run.sort()
        order = [p for p in (section_order.get(goal.id) or [])
                 if p in runs]
        order += [p for p in runs if p not in order]

        chosen: list[int] = []
        spent = 0
        for position in order:
            for index in runs[position]:
                if spent >= share:
                    break
                chosen.append(index)
                spent += len(pool[index])
            if spent >= share:
                break
        chosen.sort()
        for position in order:
            leftovers.setdefault(goal.id, []).extend(
                i for i in runs[position] if i not in chosen)
        taken.update(chosen)
        written_chars += spent
        sections.append(Section(
            heading=section_heading(goal), purpose="core",
            requirement={
                "kind": "knowledge",
                "phrase": goal.question or goal.topic,
                "terms": list(dict.fromkeys(
                    _words(goal.question) + _words(goal.topic))),
                "goal_id": goal.id,
            },
            material=[pool[index] for index in chosen],
        ))

    # a floor the sections together must clear: the goals that still
    # hold evidence take turns, so the length never comes from one part
    spare = (budget.minimum + reserve) - written_chars
    for goal in goals:
        if spare <= 0:
            break
        for index in leftovers.get(goal.id, []):
            if spare <= 0:
                break
            if index in taken:
                continue
            taken.add(index)
            spare -= len(pool[index])
            for section in sections:
                if (section.requirement or {}).get("goal_id") == goal.id:
                    section.material.append(pool[index])
                    break

    # the floor still open: validated evidence no single goal claimed
    # goes to the parts in turn — it is the request's subject, so it
    # belongs to the document even where no one goal singled it out
    if spare > 0:
        content = [s for s in sections if s.purpose == "core"]
        cycle = 0
        for index in range(len(pool)):
            if spare <= 0:
                break
            if index in taken:
                continue
            taken.add(index)
            spare -= len(pool[index])
            if content:
                content[cycle % len(content)].material.append(pool[index])
                cycle += 1
        source_order = {text: i for i, text in enumerate(pool)}
        for section in content:
            section.material.sort(key=lambda text: source_order.get(text, 0))

    if getattr(structured, "has_requirement", None) and \
            structured.has_requirement("examples"):
        examples = [i for i, sentence in enumerate(pool)
                    if i not in taken and
                    any(m in sentence.lower() for m in _EXAMPLE_MARKERS)]
        chosen = []
        spent = 0
        for index in examples:
            if spent >= 450:
                break
            chosen.append(index)
            spent += len(pool[index])
        taken.update(chosen)
        sections.append(Section(
            heading="Examples", purpose="examples",
            requirement={"kind": "examples", "phrase": "examples",
                         "terms": ["example"]},
            material=[pool[index] for index in sorted(chosen)],
        ))

    sections.append(Section(heading="_summary", purpose="summary"))
    if getattr(plan, "evidence", None) or \
            (getattr(evidence, "sources", None)):
        sections.append(Section(heading="_sources",
                                purpose="attribution"))
    return sections

def _audience_clause(value: str) -> str:
    """The shaping the request stated about its reader, phrased as
    the response's own promise rather than echoed as a label."""
    text = str(value or "").strip()
    if not text:
        return ""
    low = text.lower()
    if re.search(r"(able|ible|ly|friendly|clear|simple|short|"
                 r"concise|accessible)$", low) or \
            low in ("beginner", "expert", "advanced"):
        return f"It is written to be {text}."
    return f"It starts from {text}."


def _document_framing(purpose: str, structured, plan, state,
                      language: str) -> str:
    """The document's own opening and close: NONE are written.

    The user-visible defect this removes: sentences that describe the
    REQUEST and the RESPONSE PROCESS ("'X'의 요청 범위를 …나눠
    정리해서 설명드립니다", "위 항목을 이어서 보면 …모두 다뤘습니다")
    were realized as response content.  A response is about its
    SUBJECT: the claims carry what the sources state and the
    attribution block carries where they came from, so neither the
    request's scope nor the response's coverage is announced."""
    return ""


def generate_document_form(budget: ResponseBudget, sections: list[Section],
                           structured, plan, state: GenerationState,
                           language: str = "en") -> tuple:
    """Assemble the document response: every planned section that holds
    validated evidence is written in its dependency order; a section the
    evidence never reached is deferred (recorded, not announced), so the
    response answers what it verified and says nothing about what it
    did not."""
    language = str(language or getattr(plan, "requested_language", "")
                   or "en")
    target = str(getattr(plan, "target", "") or "")
    state.terminology.setdefault("__target__", target)
    used: set[str] = {_normalize(u) for u in state.used_evidence}

    written: list[tuple[Section, list[str]]] = []
    deferred: list[str] = []
    requirements: list[dict] = []
    material_requirements: list[str] = []
    for section in sections:
        if section.purpose in ("framing", "summary", "attribution"):
            continue
        body: list[str] = []
        for sentence in section.material:
            key = _normalize(sentence)
            if not key or key in used:
                continue
            used.add(key)
            body.append(sentence)
        # The section's material is COMPOSED, never concatenated: the
        # raw evidence sentences are cleaned, meta sentences dropped,
        # near-duplicates removed, topic-ranked and condensed before
        # they become a paragraph (brain/evidence_compose.py).
        from brain.evidence_compose import compose_body
        composed = compose_body(
            [{"text": s, "source": ""} for s in body],
            question=str(getattr(state, "request", "") or ""),
            language="ko" if str(language or "").startswith("ko")
            else "en",
            max_chars=max(1800, int(getattr(budget, "maximum", 2500)
                                     or 2500)),
            max_words=80)
        cleaned = composed.get("body") or ""
        body = [cleaned] if cleaned else []
        if body:
            for item in (composed.get("kept") or []):
                text = item.get("text") if isinstance(item, dict) else ""
                if text:
                    state.used_evidence.append(text)
        if not body:
            deferred.append(section.heading)
            continue
        written.append((section, body))
        if section.requirement:
            requirements.append(dict(section.requirement))
            material_requirements.append(
                str(section.requirement.get("phrase", "")))

    planned = [s.heading for s in sections if s.purpose != "framing"]
    state.generated_artifacts["planned_sections"] = planned
    state.generated_artifacts["requirements"] = requirements
    state.generated_artifacts["material_requirements"] = \
        material_requirements
    state.generated_artifacts["deferred_requirements"] = deferred
    state.unresolved_requirements = []
    state.generated_artifacts["outline"] = [
        {"heading": s.heading, "purpose": s.purpose,
         "goal_id": (s.requirement or {}).get("goal_id", ""),
         "deferred": s.heading in deferred}
        for s in sections]

    # The part structure is shown only for a document that TREATS
    # several parts; a single-part request is one flow — announcing its
    # own wording back would be echoing the request, not answering it.
    core_count = len([s for s in sections if s.purpose == "core"])
    request_norm = " ".join(str(getattr(state, "request", "") or "")
                            .lower().split()).rstrip("?!.")
    chunks: list[str] = []
    acc = 0
    for section, body in written:
        block = " ".join(body)
        heading_norm = " ".join(str(section.heading or "")
                                .lower().split()).rstrip("?!.")
        show_heading = (core_count >= 2 and section.purpose == "core"
                        and heading_norm
                        and heading_norm != request_norm)
        for index, paragraph in enumerate(_paragraphs(block) or [block]):
            chunks.append(f"{section.heading}\n{paragraph}"
                          if index == 0 and show_heading else paragraph)
        acc += len(block) + (len(section.heading) + 1
                             if show_heading else 0) + 4
        state.completed_sections.append(section.heading)
        state.active_section = section.heading

    parts: list[str] = []
    opening = _document_framing("framing", structured, plan, state,
                                language)
    if opening:
        parts.append(opening)
    parts.extend(chunks)
    summary = _document_framing("summary", structured, plan, state,
                                language)
    if summary:
        state.completed_sections.append("_summary")
        parts.append(summary)
    if "_sources" in planned:
        state.completed_sections.append("_sources")
    parts.extend(_sources_block(plan, language))

    text = "\n\n".join(p for p in parts if p) if parts else ""
    if text and text[-1] not in _TERMINAL:
        text += "."
    if not chunks:
        shortfall = (f"'{target}'에 대한 근거를 확보하지 못해, "
                     f"이번 응답은 여기까지만 씁니다."
                     if str(language or "").startswith("ko") else
                     f"I could not gather enough on '{target}' to "
                     f"write this up in depth, so this response stops "
                     f"here.")
        text = shortfall
    text = _enforce_maximum(text, budget, sections, plan, state, language)
    report = verify_long_form(text, state, budget, plan)
    return text, state, report


def revise_document_form(text: str, state: GenerationState,
                         budget: ResponseBudget, plan, report: dict,
                         language: str = "en") -> tuple:
    """The document's bounded revision: sentences spent twice are
    dropped and the response is measured again.  Nothing is invented
    and nothing is announced as missing — a part the evidence never
    reached was already deferred in the plan."""
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in (text or "").split("\n\n"):
        if not paragraph.strip():
            continue
        sentences = _split_sentences(paragraph)
        kept_sentences: list[str] = []
        for sentence in sentences:
            key = _normalize(sentence)
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            kept_sentences.append(sentence)
        if kept_sentences:
            kept.append(" ".join(s.strip() for s in kept_sentences))
    rebuilt = "\n\n".join(kept)
    if rebuilt and rebuilt[-1] not in _TERMINAL:
        rebuilt += "."
    report = verify_long_form(rebuilt, state, budget, plan)
    return rebuilt, state, report
