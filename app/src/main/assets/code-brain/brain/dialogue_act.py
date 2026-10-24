"""Dialogue act layer: the utterance's ILLOCUTIONARY FORCE and the
intent derived from it.

Separation of concerns (the core rule of this module):

  PREDICATE semantics decide the ACT      — what the speaker is doing
                                            (requesting, querying,
                                            asserting, expressing)
  TOPIC   semantics decide the CONTENT    — what it is about
                                            (programming, Python, Linux)

The intent type is a function of the ACT and the CONTENT's *kind*.
The TOPIC itself NEVER decides the intent type: "programming" may appear
in a request, a query, an assertion or a social remark, and each yields a
different intent.

The predicate classes below are a LEXICON OF PREDICATES (grammar): they
classify the sentence's main predicate by its morphological realization.
They are not a table mapping sentences to canned answers, and they never
mention a topic.

Derived intent types
--------------------
GREETING, CONVERSATION, INFORMATION_QUERY, EXPLANATION_REQUEST,
CAPABILITY_QUERY, LANGUAGE_CAPABILITY_QUERY, RECOMMENDATION_REQUEST,
COMMAND_QUERY, CODE_GENERATION, CODE_MODIFICATION, PROJECT_GENERATION,
ANALYSIS, CLARIFICATION
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    GREETING = "greeting"
    CONVERSATION = "conversation"
    INFORMATION_QUERY = "information_query"
    EXPLANATION_REQUEST = "explanation_request"
    # An explanation whose REQUESTED REGISTER is simplification: the
    # same target, stated for understanding rather than for detail.
    # Derived from the act's style adverbial — never from the topic.
    SIMPLIFY_EXPLANATION = "simplify_explanation"
    CAPABILITY_QUERY = "capability_query"
    LANGUAGE_CAPABILITY_QUERY = "language_capability_query"
    RECOMMENDATION_REQUEST = "recommendation_request"
    # The addressee is asked to PROVIDE an ordered way of doing
    # something: the request's own structure (a how-to clause, or a
    # focused complement headed by the procedure frame) decides this —
    # never the activity the complement names.
    PROCEDURAL_INSTRUCTION = "procedural_instruction"
    # The addressee is asked to FIND OUT: the investigation's own
    # predicate decides this, never the topic words.
    RESEARCH = "research"
    COMMAND_QUERY = "command_query"
    CODE_GENERATION = "code_generation"
    CODE_MODIFICATION = "code_modification"
    PROJECT_GENERATION = "project_generation"
    ANALYSIS = "analysis"
    CLARIFICATION = "clarification"
    # The addressee is asked to FIND WHAT IS WRONG: the investigation's
    # predicate is a FAILURE ("why does this error?"), so the turn wants
    # a repair, never a fact — the predicate's class decides it, never
    # the topic words.
    REPAIR_REQUEST = "repair_request"
    # The addressee is asked to TAKE A STANCE on a proposition the
    # utterance ITSELF carries: the act's interrogative predicates the
    # addressee's agreement, and the turn's own statement supplies what
    # is to be agreed with.  The reply discusses the claim from
    # evidence — never a definition of the topic it mentions.
    OPINION_REQUEST = "opinion_request"
    # The addressee is asked to PRODUCE a candidate: the act's frame
    # delivers content and its complement is an INDEFINITE candidate
    # with no topic to retrieve — an option the addressee must supply,
    # never a fact about anything.
    PROPOSAL_REQUEST = "proposal_request"


# The utterance's form (grammatical sentence type)
FORM_INTERROGATIVE = "interrogative"
FORM_IMPERATIVE = "imperative"
FORM_DECLARATIVE = "declarative"

# The illocutionary force (what the speaker is doing)
FORCE_REQUEST = "request"     # the speaker asks the addressee to ACT
FORCE_QUERY = "query"         # the speaker asks for INFORMATION
FORCE_ASSERT = "assert"       # the speaker states something
FORCE_EXPRESS = "express"     # the speaker performs a social act
FORCE_UNKNOWN = "unknown"

# Predicate classes — the LEXICON of predicate morphemes (grammar).
# A predicate class describes the SHAPE of the predication, never its
# topic.
_PREDICATE_CLASSES: dict[str, tuple[str, ...]] = {
    # the speaker performs a social act
    "social": (
        "hello", "hi", "hey", "thanks", "thank", "goodbye", "bye",
        "안녕", "반가워", "고마워", "고맙", "감사", "잘가", "수고",
    ),
    # ability of an addressee
    "ability": (
        "can you", "could you", "are you able", "able to", "manage to",
        "do you support", "do you handle", "do you provide",
        "할 수 있", "할수 있", "할 줄", "가능하",
    ),
    # deontic advice / recommendation
    "advice": (
        "suggest", "recommend", "advise", "propose", "should", "ought to",
        "had better", "what do you think", "which would you",
        "not sure where", "where to begin", "where do i start",
        "how do i start", "what should i", "where should i",
        "what could i", "which could i",
        "조언", "추천", "권하", "좋을 것 같", "어떤 거 좋", "뭘 해야",
        "무엇을 해야", "먼저 해야", "해야 하나", "해야 할",
        "어디서부터", "뭐부터", "무엇부터", "무엇을 먼저",
    ),
    # evaluative predication (a judgment is requested)
    "evaluation": (
        "realistic", "worth", "good idea", "bad idea", "make sense",
        "makes sense", "better", "worse", "suitable", "appropriate",
        "feasible", "practical", "reasonable", "too late", "good for",
        "현실적", "현실적", "괜찮", "나을", "낫다", "좋을", "맞는",
        "가능할", "도움이 될", "효과적",
    ),
    # difference / comparison
    "comparison": (
        "differ", "different", "difference", "compare", "versus", " vs ",
        "contrast", "차이", "비교", "어떤 점이", "어디가 다른",
    ),
    # knowledge / definitional predication
    "knowledge": (
        " is ", " are ", " means ", " refer to", " refer to", "stands for",
        "defined as", "란", "이란", "이다", "이야", "예요", "무엇", "뭐야",
        "누구", "어떤 언어",
    ),
    # action / task predication (the request makes the addresdo ACT)
    "action": (
        "make", "create", "build", "write", "generate", "implement",
        "develop", "design", "add", "fix", "repair", "debug", "refactor",
        "analyze", "explain", "list", "show", "print", "calculate",
        "find", "locate", "search", "count", "sum", "average", "sort",
        "order", "arrange", "reverse", "filter", "remove", "delete",
        "discard", "deduplicate", "eliminate", "keep", "select", "pick",
        "return", "returns", "read", "reads", "store", "stores", "save",
        "compute", "computes", "determine", "detect", "identify",
        "verify", "check", "compare", "merge", "split", "group",
        "aggregate", "maximize", "minimize", "convert", "transform",
        "apply", "process", "solve",
        "만들", "생성", "작성", "구현", "짜", "고쳐", "수정", "설명",
        "보여", "출력", "계산", "작동", "찾", "정렬", "제거", "삭제",
        "필터", "반환", "읽", "저장", "처리", "검사", "비교", "합",
        "추가", "삽입", "업데이트",
    ),
    # a failure predication: the sentence predicates that something
    # DOES NOT WORK — the SHAPE of the predication, never its topic
    "failure": (
        "error", "errors", "exception", "exceptions", "traceback",
        "fail", "fails", "failed", "failure", "crash", "crashes",
        "crashed", "broken", "bug", "bugs",
        "doesn't work", "does not work", "not working", "won't work",
        "오류", "에러", "예외", "실패", "멈추", "안 되", "안돼", "안됨",
        "작동 안", "동작 안", "고장",
    ),
    # state of the addressee (phatic)
    "state": (
        "how are you", "how do you do", "how's it going", "what's up",
        "기분", "잘 지내", "지내", "어떻게 지내", "밥 먹",
    ),
}


# The elliptical suggestion request: an interrogative whose complement is
# a first-person modal over a candidate-denoting head ("What's something
# interesting I could learn?") asks the addressee to CHOOSE — an
# indirect advice request.  The interrogative form is required: an
# imperative stating the same words is a production request.
_SUGGESTION_REQUEST = re.compile(
    r"\b(?:what|which|something|anything)\b[^?]*\bi\s+"
    r"(?:could|can|should|would|might)\s+\w", re.IGNORECASE)


# The act asks the ADDRESSEE's own stance: its interrogative predicate
# is the addressee's agreement or belief.  Pure grammar — "Do you
# agree?" asks it whatever the claim in front of it happens to be.
_STANCE_REQUEST = re.compile(
    r"\b(?:do|does|did|would|will|can|could|should|shall|isn'?t|aren'?t"
    r"|don'?t|doesn'?t|didn'?t|wouldn'?t|couldn'?t|shouldn'?t|won'?t)"
    r"\s+you\s+(?:agree|disagree|believe|feel|reckon)\b"
    r"|\b(?:don'?t|wouldn'?t|couldn'?t|shouldn'?t)\s+you\s+"
    r"(?:think|agree|believe|feel)\b"
    r"|\b(?:agree|disagree)\s+with\s+me\b",
    re.IGNORECASE)

# The indefinite CANDIDATE: the complement names no topic to retrieve
# ("something interesting", "an idea") — the addressee must SUPPLY the
# option.  A complement that then picks a topic ("something about
# space") is a request for content about that topic, never an open
# proposal.
_INDEFINITE_CANDIDATE = re.compile(
    r"\b(?:something|anything|an?\s+idea|ideas?|options?|"
    r"a\s+couple\s+of\s+things|some\s+things?)\b"
    r"|뭔가|아무거나|(?:재미있는|재미난|흥미로운)\s*(?:거|것)",
    re.IGNORECASE)

_CANDIDATE_TOPIC = re.compile(
    r"\b(?:about|on|for|of|regarding|concerning|including|between|"
    r"in|from|with|versus|vs\.?)\s+[\w가-힣]",
    re.IGNORECASE)


def _stance_request(request: str) -> bool:
    """The utterance asks whether the addressee AGREES — the
    interrogative's own predicate, never the topic it mentions."""
    return bool(_STANCE_REQUEST.search(str(request or "")))


def _carries_proposition(request: str) -> bool:
    """The utterance STATES something as well as asking: a declarative
    clause beside the stance question carries the claim to discuss
    ("…because AI can write code.  Do you agree?"), as does the
    question's own complement ("Do you agree that X is true?").

    A bare agreement question with nothing to agree with states no
    claim, so there is nothing to take a stance on — the act stays
    whatever else it is."""
    text = str(request or "").strip()
    if not text:
        return False
    for clause in re.split(r"(?<=[.!?？。])\s+", text):
        piece = clause.strip()
        if not piece or "?" in piece or "？" in piece:
            continue
        if _STANCE_REQUEST.search(piece):
            continue
        if len(piece.split()) >= 3:
            return True
    complement = re.search(
        r"\b(?:think|believe|agree|feel|reckon)\s+(?:that\s+)?"
        r"([^?!？。]{3,}?)[.?!？。]*\s*$",
        text, re.IGNORECASE)
    if complement and len(complement.group(1).split()) >= 3:
        return True
    return False


def _proposal_candidate(request: str) -> bool:
    """The act's complement is an INDEFINITE candidate with no topic
    attached: the turn wants an option produced, not knowledge
    retrieved.  The complement's own shape decides this — its
    determiner and what follows it, never what the option is about."""
    match = _INDEFINITE_CANDIDATE.search(str(request or ""))
    if not match:
        return False
    tail = str(request)[match.end():]
    if "?" in tail or "？" in tail:
        return False
    return not _CANDIDATE_TOPIC.search(tail)


def candidate_subject(request: str) -> str:
    """The candidate the request asks to BE GIVEN: the indefinite
    complement itself, carried to the end of its own sentence.  The
    speaker's state that framed it ("I'm bored.") is never the subject
    of the proposal — it is why the proposal was asked for."""
    text = str(request or "")
    match = _INDEFINITE_CANDIDATE.search(text)
    if not match:
        return ""
    tail = text[match.end():]
    sentence = re.split(r"(?<=[.!?？。])\s+", tail)[0]
    subject = (match.group(0) + sentence).strip()
    return subject.rstrip(".!?？。 ").strip()[:120]


@dataclass
class SemanticAct:
    """The utterance's meaning structure at the speech-act level."""
    form: str = FORM_DECLARATIVE
    force: str = FORCE_UNKNOWN
    subject: str = ""              # first / second / third / ""
    predicate_class: str = ""      # one of _PREDICATE_CLASSES keys or ""
    desiderative: bool = False     # first-person desire
    negated: bool = False
    past: bool = False
    method_question: bool = False  # a "how to" interrogative
    explanation_request: bool = False
    social_act: str = ""           # salutation / gratitude / valediction
    # A topic reset: a dismiss predication whose object is anaphoric
    # ("forget that", "the last topic은 지워") — the speaker retires the
    # discourse's current topic, not any content word.
    topic_reset: bool = False
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "form": self.form,
            "force": self.force,
            "subject": self.subject,
            "predicate_class": self.predicate_class,
            "desiderative": self.desiderative,
            "negated": self.negated,
            "past": self.past,
            "method_question": self.method_question,
            "explanation_request": self.explanation_request,
            "social_act": self.social_act,
            "topic_reset": self.topic_reset,
            "evidence": list(self.evidence),
        }


@dataclass
class DialogueIntent:
    """The derived intent: the type plus the semantic content it is
    about.  `type` is derived from the act; `goal`/`topics` from the
    content."""
    type: IntentType = IntentType.CLARIFICATION
    goal: str = ""                 # a short semantic goal description
    topics: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    scope: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def value(self) -> str:
        return self.type.value

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "goal": self.goal,
            "topics": list(self.topics),
            "context": dict(self.context),
            "scope": dict(self.scope),
            "evidence": list(self.evidence),
            "confidence": round(float(self.confidence), 3),
        }


# ----------------------------------------------------------------------
# The social act lexicon (a bare expressive utterance)
# ----------------------------------------------------------------------

_SOCIAL_FORMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gratitude", ("thank you", "thanks", "thx", "cheers",
                   "감사", "고마워", "고맙")),
    ("valediction", ("goodbye", "good bye", "bye", "see you", "farewell",
                     "later", "잘가", "잘 가", "안녕히 가")),
    ("salutation", ("hello", "hi", "hey", "good morning",
                    "good afternoon", "good evening",
                    "안녕하세요", "안녕", "반갑")),
)

_PRONOUNS_FIRST = (
    "i ", "i'", "we ", "me ", "my ", "our ", "us ",
    "저는", "저 ", "저희", "나는", "나 ", "내 ", "제 ", "우리는",
)
_PRONOUNS_SECOND = (
    "you ", "your ", "you'", "u ", "너", "당신", "댁", "너의",
)
_DESIRE = (
    "want", "would like", "'d like", "like to", "hope", "plan to",
    "thinking about", "think about", "consider", "aim to", "wish",
    "싶", "하고 싶", "희망", "계획", "생각해", "고려해",
)


def requested_style(text: str) -> str:
    """The REGISTER the request asks for: simple (a simplifying
    reframe) / detailed, or "" when it asks for none.

    The style is the utterance's own adverbial framing — a property of
    HOW the speaker wants it said.  It never says what the request is
    ABOUT, so it can refine an explanation into a simplifying one
    without ever entering the topic or the retrieval query.
    """
    m = _STYLE_REQUEST.search((text or "").strip())
    if not m:
        return ""
    token = (m.group(1) or "").lower()
    if token in _DETAILED_MARKERS:
        return "detailed"
    return "simple"


_STYLE_REQUEST = re.compile(
    r"((?:in\s+an?\s+(?:simpler|simple|plainer|clearer|easier)\s+way)"
    r"|(?:in\s+(?:more\s+)?detail)|(?:more\s+detailed)"
    r"|simpler|simplest|simple\s+terms|simply"
    r"|in\s+plain\s+(?:terms|english)|plain\s+english"
    r"|briefly|concisely|shortly|in\s+short|in\s+a\s+nutshell"
    r"|쉽게|간단히|간단하게|요약하면|짧게|자세하게|상세하게|더\s*쉽게)",
    re.IGNORECASE)

_DETAILED_MARKERS = {"in detail", "more detailed", "자세하게", "상세하게"}


def analyze_act(text: str, lang: str = "en") -> SemanticAct:
    """Derive the utterance's speech-act structure.

    Purely grammatical: sentence form, negation, tense, subject person,
    the predicate's class, and whether the predication is a desire.
    """
    act = SemanticAct()
    raw = text.strip()
    lower = raw.lower()
    if not lower:
        return act

    # ── form ──
    # A "how to" ask: the interrogative WH-form OR the nominal form
    # ("the way to list files", "파일 목록 보는 법") — both predicate a
    # METHOD, which is a grammatical property of the utterance.
    how = bool(re.search(
        r"^(how\b|어떻게)|\bhow\s+(to|do|can|would|should)\b|"
        r"(하는 법|하는법|보는 법|보는법|쓰는 법|확인하는 법)|법$|방법",
        lower))
    if "?" in raw or "？" in raw:
        act.form = FORM_INTERROGATIVE
        act.method_question = how
    elif re.match(
            r"^(what|why|when|where|who|which|whom|whose|how)\b", lower):
        act.form = FORM_INTERROGATIVE
        act.method_question = how
    elif how:
        act.form = FORM_INTERROGATIVE
        act.method_question = True
    elif _starts_with_morpheme(
            lower,
            ("만들", "생성", "작성", "구현", "짜", "고쳐", "수정해",
             "설명해", "보여줘", "출력해", "make", "create", "build",
             "write", "generate", "implement", "add", "fix", "delete",
             "remove", "explain", "show", "list", "give ", "provide ",
             "please", "plz", "pls")):
        act.form = FORM_IMPERATIVE
    elif lower.rstrip("!?.~").endswith(
            ("해줘", "해 주세요", "주세요", "줘", "십시오", "바랍니다",
             "필요해", "필요하다", "please")):
        act.form = FORM_IMPERATIVE
    else:
        act.form = FORM_DECLARATIVE

    # ── negation / tense (grammatical morphology) ──
    # Negation and tense scope over the MAIN predication.  A negator
    # inside a complement or an adverbial clause ("as if I have never
    # written code", "because it doesn't work") negates THAT clause —
    # it never negates the utterance's own act, so the request keeps
    # its force.
    main = _main_clause(lower)
    act.negated = bool(re.search(
        r"(don't|doesn't|do\s+not|didn't|not\s+to|never|isn't|aren't|"
        r"wasn't|weren't|"
        r"않다|않는|않어|않네|싶지\s*않|원하지\s*않|못\s*하)", main))
    act.past = bool(re.search(
        r"\b(was|were|did|had|have heard|'ve heard|heard)\b|"
        r"했다|였다|됐다|됬다|생성됐|만들어졌|생성되었|만들었|들었", main))

    # ── subject person ──
    if _starts_with_morpheme(lower, _PRONOUNS_FIRST) or \
            any(p in lower for p in ("i am", "i'm", "i was", "i have",
                                     "i've", "저는", "나는")):
        act.subject = "first"
    elif any(p in lower for p in _PRONOUNS_SECOND):
        act.subject = "second"

    # ── predicate class (the lexicon of predicates) ──
    act.social_act = _social_act(lower)
    # ── explanation request (a performative verb phrase) ──
    act.explanation_request = bool(
        re.match(r"^(explain|describe|talk\s+about|"
                 r"tell\s+me\s+(?:\w+\s+){0,3}about)\b", lower)
        or re.search(r"^(can|could|would)\s+you\s+(explain|describe)\b",
                     lower)
        # the method question ABOUT explaining: the activity the
        # method is for IS the explanatory act, so the turn asks for an
        # explanation, never for an ordered procedure
        or re.search(r"\bhow\s+(?:(?:do|does|did|can|could|would|should"
                     r"|will|may|might)\s+)?(?:(?:i|we|you|he|she|they|it)"
                     r"\s+)?(?:explain|describe|clarify|summar"
                     r"|rephrase|paraphrase|interpret)\b", lower)
        or re.search(r"(설명해|설명\s*해|뭐야|뭐냐|무엇인|어떤\s+의미)",
                     lower)
        or _has_comprehension_gap(lower))
    act.predicate_class = _classify_predicate(lower, act)

    # ── topic reset (a dismiss predication over an anaphoric object) ──
    act.topic_reset = _is_topic_reset_clause(lower)

    # ── desire (first-person) ──
    act.desiderative = act.subject == "first" and _starts_with_or_contains(
        lower, _DESIRE)

    # ── force ──
    act.force = _derive_force(act)

    act.evidence = [
        f"form={act.form}",
        f"predicate={act.predicate_class or 'unclassified'}",
        f"subject={act.subject or 'unspecified'}",
    ]
    if act.negated:
        act.evidence.append("negated")
    if act.past:
        act.evidence.append("tense=past")
    return act


def _starts_with_morpheme(lower: str, morphemes) -> bool:
    stripped = lower
    for marker in ("please ", "plz ", "pls "):
        if stripped.startswith(marker):
            stripped = stripped[len(marker):]
            break
    return any(stripped.startswith(m) for m in morphemes)


def _starts_with_or_contains(lower: str, morphemes) -> bool:
    return any(m in lower for m in morphemes)


def discourse_action(text: str) -> dict | None:
    """The turn's DISCOURSE operation, as a structural act.

    A dismiss predication over an anaphoric object is a discourse
    operation on the topic state — `reset_topic` scoped to the current
    topic — not a sentence to remember.  The remaining clauses of the
    utterance are then analyzed on their own; nothing about this
    operation is particular to any one phrasing (the clause's predicate
    and its anaphoric object decide it).
    """
    if not _is_topic_reset_clause((text or "").lower()):
        return None
    return {
        "type": "reset_topic",
        "scope": "current_topic",
        "evidence": ["dismiss predicate over an anaphoric object"],
    }


def separate_topic_reset(text: str) -> tuple[str, bool]:
    """Split a topic-reset act off the utterance: the dismiss clause
    retires the discourse's topic, the REMAINING clauses are the
    utterance's live content ("Actually, forget the last topic. Tell me
    about space." → "Tell me about space.").  Returns
    (remaining_text, had_reset)."""
    if not _is_topic_reset_clause(text.lower()):
        return text, False
    clauses = re.split(r"(?<=[.!?？。])\s+", text.strip())
    kept = [c for c in clauses if c.strip()
            and not _is_topic_reset_clause(c.lower())]
    if kept:
        return " ".join(kept), True
    # A COORDINATED reset: the dismiss predicate and the utterance's
    # live content share one clause ("그 얘기는 잊고 우주에 대해
    # 알려줘").  The dismiss scope ends at its own predicate; what is
    # coordinated with it — a clause carrying its own request — is the
    # utterance's content and stays.
    for clause in clauses:
        remainder = _after_dismiss_scope(clause)
        if remainder:
            return remainder.strip(), True
    return "", True


def _after_dismiss_scope(clause: str) -> str:
    """The live content coordinated with a dismiss predicate in the SAME
    clause, or "".

    The dismiss scope is its predicate plus its object ("forget the
    last topic" — the object is what is dismissed).  Whatever is
    coordinated with it and still CARRIES A REQUEST is the utterance's
    own content, never part of the operation.
    """
    lower = clause.lower()
    for d in _DISMISS:
        idx = lower.find(d)
        if idx == -1:
            continue
        after = clause[idx + len(d):].strip()
        # the object the dismiss act takes, when it follows its verb
        obj = re.match(
            r"^(?:that|it|this|them|the\s+last\s+topic|that\s+topic|"
            r"previous\s+topic|그\s*얘기[는]?|그거|그것|그건|아까)"
            r"[\s,]*", after, flags=re.IGNORECASE)
        if obj:
            after = after[obj.end():].strip()
        # the coordinating suffix / conjunction the two clauses share
        after = re.sub(r"^(?:고|며|서|지만)\s+", "", after.strip())
        after = re.sub(r"^(?:,\s*|\s+and\s+|\s+then\s+)",
                       "", after.strip(), flags=re.IGNORECASE)
        if after and (_content_delivery_request(after)
                      or _is_request(after)):
            return after
        return ""
    return ""


def _ko_anaphoric_object(clause: str) -> bool:
    """Whether the clause's object is a DISCOURSE referent — the
    anaphoric object a dismiss act requires.  Either a demonstrative
    pronoun ("그거") or a demonstrative determiner over a discourse noun
    ("그 얘기는" — that story)."""
    for a in ("그거", "그것", "아까", "그건", "그얘기", "그 얘기",
              "그 이야기"):
        if a in clause:
            return True
    for m in re.finditer(r"(?:그|이|저)\s+([가-힣]+)", clause):
        noun = m.group(1).rstrip("는은이가을를의도만좀")
        if not noun:
            continue
        if any(noun.startswith(d) for d in _DISCOURSE_NOUNS
               if len(d) >= 2):
            return True
    return False


def _is_topic_reset_clause(lower: str) -> bool:
    """A clause retires the discourse's topic when its predicate is a
    dismiss verb and its object is the anaphor ("forget that", "forget
    the last topic").  An operation object ("drop duplicates") is never
    a reset — the object must be the anaphoric reference itself.  The
    check runs per clause: a reset instruction inside a larger
    utterance retires only its own scope."""
    import re as _re
    for clause in _re.split(r"[.!?\n。？]", lower):
        clause = clause.strip()
        if not clause or not any(d in clause for d in _DISMISS):
            continue
        for d in _DISMISS:
            idx = clause.find(d)
            if idx == -1:
                continue
            after = clause[idx + len(d):].strip(" ,")
            if any(after.startswith(a + " ") or after == a
                   for a in ("that", "it", "this", "them",
                             "그거", "그것")):
                return True
        if any(p in clause for p in _TOPIC_ANAPHORA_PHRASES):
            return True
        # Korean: the object may precede the dismiss verb ("그 얘기는
        # 잊고 …"), and it is anaphoric when a demonstrative names a
        # discourse referent.
        if _ko_anaphoric_object(clause):
            return True
    return False


def _has_comprehension_gap(lower: str) -> bool:
    """A first-person comprehension-gap predication ("I don't really
    understand why...") — the speaker asks for the gap to be filled:
    an explanation request by its own predication.  The negation is
    the speaker's STATE, not the request's force ("don't explain X"
    negates the request; "I don't understand X" motivates it)."""
    return bool(re.search(
        r"(don't|do not|can't|cannot|didn't)\s+(really\s+|quite\s+|"
        r"fully\s+)?(understand|get|grasp|follow)|"
        r"이해가 안|이해를 잘 못|모르겠", lower))


# The embedded interrogative a comprehension gap carries: the speaker
# states WHICH proposition they fail to understand, and that complement
# — not a domain noun occurring inside it — is the utterance's focused
# content ("I don't understand why languages need types" asks about
# "languages need types").
_GAP_WH = re.compile(
    r"\b(?:understand|get|grasp|follow|see|realize|work\s+out|"
    r"이해|알지)\s+"
    r"(?:why|what|how|when|where|whether|which|왜|무엇|어떻게|언제|어디)"
    r"\s+(.+)$", re.IGNORECASE)


def _gap_complement(lower: str) -> str:
    """The complement of the utterance's embedded interrogative when the
    utterance predicates a comprehension gap; "" otherwise."""
    if not _has_comprehension_gap(lower):
        return ""
    m = _GAP_WH.search(lower)
    if not m:
        return ""
    return m.group(1).strip().rstrip(".?!~\u3002").strip()


# The clause boundary: a subordinator or a complementizer introduces a
# clause of its OWN — its negation, tense and modality belong to that
# clause.  Grammar, never a topic list.
_MAIN_CLAUSE_BOUNDARY = re.compile(
    r"\s+(?:as\s+if|as\s+though|because|since|although|though|unless|"
    r"whereas|while|when|after|before|until|so\s+that|in\s+order\s+that|"
    r"if|whether|that\s+(?:i|we|you|he|she|they|it)|"
    r"given\s+that|now\s+that)\b|"
    r"[,;]\s*(?:because|since|although|though|unless|if|whether|as\s+if)\b|"
    r"(?:때문에|기\s*때문에|면서|는데|은데|다면|면|으면서|어서|아서\s)",
    re.IGNORECASE)


# A sentence boundary the utterance actually continues past: the period
# of an abbreviation or a decimal never opens a new predication, so the
# cut is taken only where a sentence really ends.
_SENTENCE_BOUNDARY = re.compile(r"[.!?。！？]+(?=\s|$)")


def _first_sentence(text: str) -> str:
    """The utterance's first predication: later sentences elaborate,
    they never change what THIS sentence asserts or denies."""
    match = _SENTENCE_BOUNDARY.search(text or "")
    if not match:
        return text or ""
    return text[:match.end()].strip() or text


def _main_clause(text: str) -> str:
    """The utterance's own predication: its first sentence, cut again at
    the first clause boundary inside it.  The subordination's morphology
    is read inside the subordinated clause, never as the utterance's
    own; and a later sentence ("I have never written code before") is
    the speaker's background, never this turn's request."""
    sentence = _first_sentence(text or "")
    match = _MAIN_CLAUSE_BOUNDARY.search(sentence)
    if not match:
        return sentence
    return sentence[:match.start()] or sentence


def _classify_predicate(lower: str, act: SemanticAct) -> str:
    """Classify the sentence's main predicate by its morphological
    realization.  The predicate class describes the SHAPE of the
    predication — never its topic."""
    # Order matters: a social remark inside a larger sentence is still
    # social only when the utterance IS the social act.
    if act.social_act and act.form != FORM_INTERROGATIVE:
        return "social"
    # An ability predicate addressed at the addressee
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["ability"]):
        # "Can you make X" is a request for action, not an ability check.
        # Only a question that DIRECTLY addresses the addressee with the
        # ability modal is promoted; a WH-question about ability
        # ("What can you help me with?") stays an ability query.
        if act.form == FORM_INTERROGATIVE and \
                _starts_with_or_contains(lower, _PREDICATE_CLASSES["action"]) \
                and not _starts_with_or_contains(
                    lower, _PREDICATE_CLASSES["advice"]) \
                and re.match(r"^(can you|could you|will you|would you|"
                             r"할 수|할수)", lower) \
                and not act.explanation_request \
                and not _mentions_addressee_language(lower):
            return "action"
        return "ability"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["state"]):
        return "state"
    if act.form == FORM_INTERROGATIVE and _SUGGESTION_REQUEST.search(lower):
        # The elliptical suggestion request: a WH-question whose
        # complement is a first-person modal asks the addressee to pick a
        # CANDIDATE ("What's something I could learn this weekend?") — an
        # indirect advice request, not a definitional query.
        return "advice"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["advice"]):
        return "advice"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["evaluation"]):
        return "evaluation"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["comparison"]):
        return "comparison"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["action"]):
        return "action"
    # A failure predication is classified among the predicate classes
    # AFTER an action: an utterance that also predicates an action
    # ("fix the error") is an action request whose object is the
    # failure.  The failure class claims the utterances that predicate
    # nothing else — the sentence only states that something does not
    # work.
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["failure"]):
        return "failure"
    if _starts_with_or_contains(lower, _PREDICATE_CLASSES["knowledge"]):
        return "knowledge"
    return ""


def _mentions_addressee_language(lower: str) -> bool:
    """The predicate's object is a natural language (the language
    capability topic)."""
    return bool(re.search(
        r"\b(english|korean|japanese|chinese|spanish|french|german|"
        r"italian|russian|portuguese)\b|"
        r"영어|한국어|일본어|중국어|스페인어|프랑스어|독일어", lower))


def _social_act(lower: str) -> str:
    """A BARE expressive utterance's act (salutation/gratitude/valediction).

    Only when the utterance IS the social act: any substantial remainder
    means the greeting is a component of a larger sentence and the whole
    sentence is classified by its own predicate.
    """
    u = lower.strip().rstrip("!?.~")
    if not u:
        return ""
    for act_name, forms in _SOCIAL_FORMS:
        for form in forms:
            if u == form:
                return act_name
            if u.startswith(form):
                rest = u[len(form):]
                if not rest:
                    return act_name
                if form.isascii():
                    if rest[0] in " ,!?'\"-":
                        remainder = rest.lstrip(" ,!?'\"-").strip()
                        if not _has_substantial_content(remainder):
                            return act_name
                else:
                    remainder = rest.strip()
                    if not _has_substantial_content(remainder):
                        return act_name
    return ""


def _has_substantial_content(remainder: str) -> bool:
    """True when the words after the social marker carry independent
    propositional content (a clausal remainder)."""
    if not remainder:
        return False
    # A remainder with a finite clause (a subject + a verb phrase, a
    # question, or a desire statement) is substantial content.
    if "?" in remainder:
        return True
    if re.search(
            r"\b(i|we|you|he|she|they|it|i'm|i've|i'll|we're|you're|"
            r"i'd|that|this|there)\b", remainder):
        return True
    if len(remainder.split()) >= 4:
        return True
    return False



def _derive_force(act: SemanticAct) -> str:
    """The illocutionary force from the grammatical structure."""
    if act.social_act and act.form == FORM_DECLARATIVE and \
            act.predicate_class == "social":
        return FORCE_EXPRESS
    if act.explanation_request and not (act.negated or act.past):
        # An explanation performative asks for an EXPLANATION of a
        # meaning — never for an artifact, whatever the sentence form.
        return FORCE_QUERY
    if act.form == FORM_IMPERATIVE and not (act.negated or act.past):
        return FORCE_REQUEST
    if act.form == FORM_INTERROGATIVE:
        # A polite action request phrased as a question: "Can you make a
        # parser?" — the speaker asks the addressee to ACT.
        if act.predicate_class == "action" and \
                act.subject == "second":
            return FORCE_REQUEST
        return FORCE_QUERY
    # A declarative that requests action through a politeness formula
    # ("I need you to make...", "정확히 뭐가 필요해" with an action
    # predicate) is still a request when it predicates the addressee's
    # action.
    if act.predicate_class == "action" and act.subject == "second":
        return FORCE_REQUEST
    if act.form == FORM_DECLARATIVE:
        return FORCE_ASSERT
    return FORCE_UNKNOWN


# ----------------------------------------------------------------------
# Topic extraction and anaphora resolution (content, not act)
# ----------------------------------------------------------------------

_QUESTION_FRAMING = re.compile(
    r"^(?:(?:what|who|whom|whose|which|how|why|when|where)\b"
    r"(?:\s*'(?:s|re)\b|\s+(?:is|are|was|were|do|does|did|can|could|"
    r"would|will|should|to|going|it|that|this|the|a|an|am))?"
    r"(?:\s+(?:i|you|we|they|he|she|it))?"
    r"\s+)",
    re.IGNORECASE)

# A leading performative marker is grammar, never content: strip it
# before reading the focused constituent.
_LEADING_PERFORMATIVE = re.compile(
    r"^(?:please\s+|plz\s+|pls\s+|let's\s+|lets\s+)?"
    r"(?:explain|describe|talk\s+about|discuss|"
    r"tell\s+me\s+(?:\w+\s+){0,3}about|make|create|build|write|"
    r"generate|implement|develop|design|add|fix|repair|debug|refactor|"
    r"find|locate|list|show|print|calculate|remove|delete|discard|"
    r"compare|contrast|check|verify|determine|detect|identify|"
    r"sort|filter|keep|select|return|store|save|read|process|"
    r"만들|생성|작성|구현|설명|찾|정렬|제거|삭제|출력|계산)\s+",
    re.IGNORECASE)

# Cut the focused constituent off at the subject/auxiliary that follows
# it: "language would you suggest I start with" → "language".
_TOPIC_CUT = re.compile(
    r"\s+(?:would you|could you|should i|should we|will you|can i|"
    r"can you|do i|do you|did you|are you|is it|are they|should we|"
    r"is|are|was|were|am|do|does|did|can|could|would|will|should|"
    r"i|you|we|they|me|to|the|a|an)\b.*$",
    re.IGNORECASE)

_CLAUSE_SPLIT = re.compile(r"(?<=[.!?？。])\s+")

_ANAPHORA = ("they", "them", "their", "these", "those", "both",
             # The singular anaphor class: a demonstrative/personal
             # pronoun whose antecedent lives in the previous discourse
             # (an anaphor precedes its antecedent — the discourse, not
             # the sentence, supplies it).
              "that", "it", "this", "him", "her",
              "그것", "그거", "그거는", "그것은", "그걸",
              "그걸로", "아까", "그거을",
              # The Korean demonstratives: written as one word or as a
              # determiner over a discourse noun, they reach back the
              # same way the English ones do.
              "그건", "그얘기", "그 얘기", "그 이야기", "이 얘기는")

# The dismiss predicate class (grammar): a clause that retires the
# discourse's current topic.  The OBJECT must be anaphoric — "drop
# duplicates" predicates an operation, never a topic reset.  The Korean
# entries are STEMS so the conjunctive forms ("잊고" — "forget and…")
# match the same predicate.
_DISMISS = ("forget", "dismiss", "drop that", "drop it", "clear that",
            "clear it", "skip that", "skip it", "move on from",
            "잊어버려", "잊어", "잊고", "잊", "지워", "지우고", "지우",
            "버려", "버리고", "그만하", "접어")

# The discourse nouns a demonstrative may be the object of: "그 얘기는"
# (that story), "그 주제" (that topic) — the object of a dismiss act
# must be a DISCOURSE referent, never an operation's patient.
_DISCOURSE_NOUNS = ("topic", "story", "discussion", "subject",
                    "얘기", "이야기", "주제", "말", "얘", "거", "것")

# A phrase-level anaphor: the topic itself referred to by name.
_TOPIC_ANAPHORA_PHRASES = ("last topic", "that topic", "the topic",
                           "previous topic", "이전 주제", "아까 주제")

# The demonstrative's distribution (grammar).  It stands in two
# positions:
#
#   DETERMINER — "this weekend", "that day": the demonstrative heads a
#       nominal the UTTERANCE itself supplies — a time expression
#       anchored to the speech situation (see _DEICTIC_ANCHORS below),
#       or a closed-class complement — so nothing in it reaches back to
#       the discourse.
#
#   ANAPHOR — "explain that", "give me that", "explain that more
#       simply": nothing the utterance supplies completes the
#       demonstrative (only closed-class words or predicate modifiers
#       follow it), so its referent is the DISCOURSE's.
#
# A nominal that is not a deictic anchor is itself the referent — "that
# report" points back at a report the discourse mentioned; only an
# utterance-anchored nominal ("this weekend") makes the demonstrative
# a mere determiner.
_DEMONSTRATIVE_CLOSED = frozenset("""
in on at to for with of from by about as into onto over under between
during before after through without within against along across behind
below beside beyond near since toward towards until upon via per than
despite among around and or but nor so yet because although while if
unless whether then else than not
is are was were be been being am do does did done have has had
will would shall should can could may might must
i you he she we they me him her us them my your his its our their
a an the this that these those some any no every each another other
whose which what who whom
again soon later already still just now then here there once twice
always never really simply clearly actually basically exactly together
alone well fine
다음 전에 후에 부터 동안 그리고 그러나 하지만 때문에 만약
""".split())

# The nominals a demonstrative modifies DEICTICALLY: time expressions
# anchored to the utterance's own situation ("this weekend", "that
# day").  The utterance supplies their referent — they never point back
# at the discourse, so a demonstrative heading one is a determiner, not
# an anaphor.  A nominal OUTSIDE this class is a referent the discourse
# supplies and the demonstrative reaches for it.
_DEICTIC_ANCHORS = frozenset("""
today tomorrow yesterday tonight noon midnight
weekend weekends weekday weekdays week weeks month months year years
day days morning mornings afternoon evenings evening night nights
hour hours minute minutes second seconds moment moments
time times period periods season seasons summer winter spring autumn
fall decade centuries century date
""".split())

_ANAPHOR_RE = re.compile(
    r"\b(" + "|".join(a for a in _ANAPHORA if a.isascii()) + r")\b")

_KO_ANAPHORA = tuple(a for a in _ANAPHORA if not a.isascii())


def _anaphor_hits(lower: str) -> list[tuple[int, str]]:
    """The clause's anaphoric references with their positions.

    A demonstrative that heads a deictically anchored nominal ("this
    weekend", "that day") is excluded: the nominal names a time the
    utterance itself supplies, so nothing in the phrase reaches back.
    Every other demonstrative — standing alone, followed by a
    closed-class word or by a predicate modifier ("explain that more
    simply") — is an anaphor whose referent the discourse supplies."""
    hits: list[tuple[int, str]] = []
    for m in _ANAPHOR_RE.finditer(lower):
        word = m.group(1)
        if word in ("this", "that"):
            rest = lower[m.end():].lstrip(" \t,;:'\"\u2014-")
            nxt = re.match(r"[A-Za-z']+", rest)
            if nxt and nxt.group(0).lower() not in _DEMONSTRATIVE_CLOSED \
                    and nxt.group(0).lower() in _DEICTIC_ANCHORS:
                # a determiner over a nominal the utterance supplies
                continue
        hits.append((m.start(), word))
    for a in _KO_ANAPHORA:
        start = 0
        while True:
            i = lower.find(a, start)
            if i < 0:
                break
            hits.append((i, a))
            start = i + len(a)
    hits.sort()
    return hits

# Words that are never topics (greetings, discourse markers, WH-words)
_NON_TOPIC = {
    "i", "we", "you", "they", "it", "my", "your", "our", "the", "a", "an",
    "this", "that", "these", "those", "what", "how", "why", "when",
    "where", "which", "who", "whom", "whose", "is", "are", "was", "were",
    "do", "does", "did", "can", "could", "would", "should", "will", "am",
    "hi", "hello", "hey", "thanks", "thank", "please", "yes", "no", "ok",
    "okay", "also", "then", "but", "and", "or", "if", "so", "now", "just",
    "really", "very", "much", "many", "some", "any", "me", "us", "him",
    "her", "them", "his", "its", "been", "have", "has", "had", "will",
    "make", "create", "build", "write", "need", "want", "like",
    "begin", "start", "focus", "hear", "heard", "think", "thinking",
    "sure", "experience", "beginner", "realistic", "language",
    "given", "longest", "function", "routine", "program", "code",
    "list", "numbers", "number", "value", "values", "sum", "zero",
    "maximum", "minimum", "average", "duplicates", "duplicate",
    "remaining", "descending", "ascending", "order", "integer",
    "integers", "string", "strings", "element", "elements", "file",
    "files", "sort", "find", "explain", "describe", "remove", "delete",
    "compare", "contrast", "check", "verify", "determine", "detect",
    "identify", "apply", "transform", "convert", "merge", "split",
    "keep", "select", "return", "read", "store", "save", "process",
    "calculate", "print", "show", "locate", "input", "output", "dataset",
    "score", "student", "array", "sequence", "table", "record", "step",
    "step1", "result", "results", "data", "item", "items", "name",
    "first", "second", "third", "many", "few", "whether", "kind", "type",
    "tell", "ask", "give", "help",
}


# ── Korean topic morphology ────────────────────────────────────────
# A Hangul token names a topic through its STEM: the particle marks the
# token's grammatical role in the clause, so it is stripped before the
# token is judged.  The closed class, the request's own framing and the
# provision performative are grammar — they never name a topic.
_KO_PARTICLES = (
    "에 대해", "에 대해서", "에 대한", "으로서", "로써", "에서는",
    "에게서", "에서", "에게", "으로", "까지", "부터", "보다", "처럼",
    "만큼", "이나", "이랑", "하고", "에", "의", "은", "는", "이", "가",
    "을", "를", "도", "만", "과", "와", "로", "랑",
)

# The Korean closed class, the request's framing and the register the
# request asks for: grammar, never a topic.
_KO_NON_TOPIC = frozenset({
    # deixis / closed class
    "그", "그거", "그것", "그건", "이", "이것", "이건", "저", "그런",
    "이런", "뭐", "무엇", "어떤", "어디", "언제", "왜", "어떻게", "누구",
    "그리고", "하지만", "그냥", "또", "또는", "혹은", "수", "것", "거",
    "때", "이유", "중", "등", "및", "한", "명", "개", "좀", "더", "잘",
    # the interrogative's own predicative form: what the utterance
    # ASKS with, never what it asks about
    "뭐야", "뭐죠", "뭐니", "뭡니까", "누구야", "어디야", "언제야",
    # the predication's own copular / existential forms
    "있어", "없어", "같아", "어때", "어떠", "괜찮", "좋아", "나빠",
    "힘들", "어려워", "쉬워", "해",
    # the request's framing: WHAT KIND of content is asked for
    "예시", "예제", "예를", "예", "얘기", "이야기", "주제", "정보",
    "설명", "방법", "차이", "비교", "정답", "개념", "질문", "답변",
    # the register the request asks for
    "쉽게", "간단히", "간단하게", "자세히", "짧게", "요약하면",
    "맛있게", "맛있", "천천히", "빨리", "조용히", "꼼꼼히",
    "차근차근", "제대로", "처음부터", "끝까지",
    # politeness / social
    "안녕", "반가워", "고마워", "감사", "부탁",
})

# The provision performative stems (inform / show / explain / teach) and
# the topic-reset predicate: the token carrying one is the utterance's
# ACT, never its topic.
_KO_PROVISION_STEMS = ("알려", "보여", "설명해", "가르쳐", "들어줘",
                       "줄래", "해줘", "줘", "잊", "지워", "버려")

_HAS_HANGUL = re.compile(r"[가-힣]")


def _ko_strip_particle(token: str) -> str:
    """The token's stem: its particle removed (the longest match first,
    so "에 대해" is not cut down to a bare "에") and its attributive
    ending dropped — "요리하는" is "요리" + a relativizer, so the stem
    that names the activity is "요리"."""
    stem = token
    for particle in _KO_PARTICLES:
        if len(stem) > len(particle) and stem.endswith(particle):
            stem = stem[:-len(particle)]
            break
    # the light-verb stem: <content> + 하 ("요리하-") / 되 ("설치되-")
    while len(stem) > 2 and stem.endswith(("하", "되")):
        stem = stem[:-1]
    return stem


def _ko_non_topic(token: str) -> bool:
    """Whether the Hangul token is grammar: the closed class, the
    request's own framing or the provision performative."""
    if not _HAS_HANGUL.search(token):
        return False
    if token in _KO_NON_TOPIC or _ko_strip_particle(token) in _KO_NON_TOPIC:
        return True
    return any(stem in token for stem in _KO_PROVISION_STEMS)


def _hangul_follows(text: str, pos: int) -> bool:
    """Whether a Hangul character immediately follows `pos`.

    A sentence-initial capital is English orthography — every English
    sentence starts capitalized — so at a sentence start it names
    nothing.  When a Hangul particle attaches to the word instead
    ("Python이 뭐야"), the capitalization is not English orthography at
    all: the mixed-script token names its entity, sentence start or
    not."""
    return bool(_HAS_HANGUL.match(text[pos:pos + 1]))


def _salient_nouns(text: str) -> list[str]:
    """The content-bearing noun phrases of an utterance — the antecedent
    pool for anaphora resolution.  Proper nouns and knowledge-topic
    nouns, in order of appearance.

    A capitalized word at a SENTENCE-INITIAL position is grammar
    (English orthography capitalizes every sentence's first word), not
    proper-noun evidence — "Let's talk about..." is not about a topic
    named "Let".  Only a mid-sentence capital names an entity."""
    found: list[str] = []
    seen: set[str] = set()
    # The sentence-initial positions: the text's start and the word
    # after a sentence boundary — capitalization there is grammar.
    sentence_starts = {0}
    for m in re.finditer(r"[.!?？。]\s+", text):
        sentence_starts.add(m.end())
    # Proper nouns (a language, a framework, a domain name).  The
    # boundary is explicit so a Hangul particle attached to the word
    # ("Python이") does not swallow it: a mixed-script token's Latin
    # part still names the entity.
    for m in re.finditer(
            r"(?<![A-Za-z0-9_+#])([A-Z][A-Za-z+#]{1,20})"
            r"(?![A-Za-z0-9_+#])", text):
        w = m.group(1)
        if w.lower() in _NON_TOPIC:
            continue
        if m.start() in sentence_starts and not _hangul_follows(text,
                                                                m.end()):
            continue
        if w.lower() not in seen:
            seen.add(w.lower())
            found.append(w)
    # Common knowledge-topic nouns.  A compound names the WHOLE topic:
    # "programming languages" is one referent, not the word
    # "programming" with its noun dropped.
    for m in re.finditer(
            r"(?<![A-Za-z0-9_+#])(python|kotlin|java|javascript|"
            r"typescript|golang|go|rust|"
            r"c\+\+|ruby|swift|php|scala|haskell|"
            r"programming(?:\s+languages?)?|data structures?|"
            r"algorithms?|databases?|webs?|apps?|games?|machine learning|"
            r"mobile|android|ios|web development|data science)"
            r"(?![A-Za-z0-9_+#])",
            text, re.IGNORECASE):
        w = m.group(1)
        if w.lower() not in seen:
            seen.add(w.lower())
            found.append(w)
    # Korean: the content STEM of each Hangul token — the particle is
    # the token's grammatical role, the closed class and the request's
    # framing are grammar, and only what is left names a topic.
    if _HAS_HANGUL.search(text):
        for m in re.finditer(r"[가-힣]{2,}", text):
            token = m.group(0)
            stem = _ko_strip_particle(token)
            if len(stem) < 2 or _ko_non_topic(token) \
                    or _ko_non_topic(stem):
                continue
            key = stem.lower()
            if key not in seen:
                seen.add(key)
                found.append(stem)
    return found


def _query_clause(text: str) -> str:
    """The clause that carries the question: the interrogative clause of
    a multi-sentence utterance, else the REQUEST clause (an imperative
    or a hortative — the act lives there), else the whole utterance."""
    clauses = [c.strip() for c in _CLAUSE_SPLIT.split(text.strip())
               if c.strip()]
    if not clauses:
        return text.strip()
    for c in clauses:
        if "?" in c or "？" in c:
            return c
    for c in clauses:
        if re.match(r"^(what|who|whom|whose|which|how|why|when|where)\b",
                    c, re.IGNORECASE):
            return c
    for c in clauses:
        if _request_clause(c):
            return c
    return clauses[0]


_HORTATIVE = re.compile(r"^let'?s\b", re.IGNORECASE)


def _request_clause(clause: str) -> bool:
    """The clause CARRIES the utterance's request: an imperative or a
    hortative.  "I'm tired. Let's talk about X" states a report in its
    first clause and PROPOSES in its second — the proposal is the
    utterance's focused content, the report is not."""
    c = clause.strip()
    if not c:
        return False
    if _HORTATIVE.match(c):
        return True
    try:
        return analyze_act(c).form == FORM_IMPERATIVE
    except Exception:
        return False


# The content-delivery performatives (grammar): the request's predicate
# asks the addressee to HAND OVER content.  The frame carries no topic
# of its own — its content is whatever the DISCUSSION is about.
_CONTENT_DELIVERY = (
    "give", "gives", "giving", "gave",
    "show", "shows", "showing", "showed", "shown",
    "tell", "tells", "telling", "told",
    "send", "sends", "sending", "sent",
    "share", "shares", "sharing", "shared",
    "describe", "describes", "describing", "described",
    "explain", "explains", "explaining", "explained",
    "demonstrate", "demonstrates", "demonstrating",
    "illustrate", "illustrates", "illustrating",
    "present", "presents", "presenting",
    "offer", "offers", "offering",
    # The light verb (-해/-어 주다) carries the REQUEST's force; it does
    # not ask for content.  "X를 알려줘" asks for content through its
    # OWN performative ("알려"), while "-해줘" only makes the sentence a
    # request — which content the request carries is the clause's own
    # business.
    "보여", "알려", "설명해", "가르쳐",
)


# The self-presentation performative: the verb's OBJECT is the
# ADDRESSEE themself ("introduce yourself", "자기소개해줘", "tell me
# about yourself").  What is asked for is an act performed in the
# conversation, never a body of content about a topic.
_SELF_PRESENTATION = re.compile(
    r"\b(?:introduce|present|describe)\s+(?:yourself|themselves)\b"
    r"|\byourself\b|\byour\s+(?:name|self|introduction|background)\b"
    r"|\bwho\s+(?:are\s+you|you\s+are)\b|about\s+you\b"
    r"|자기소개|본인\s*소개|본인을|자기\s*소개|네\s*이름|너\s*이름",
    re.IGNORECASE)


def _self_presentation_request(clause: str) -> bool:
    """Whether the clause asks the addressee to present themselves."""
    return bool(_SELF_PRESENTATION.search(clause or ""))


def _content_delivery_request(clause: str) -> bool:
    """Whether the clause's frame is a request for content."""
    lower = (clause or "").lower()
    if not lower:
        return False
    for form in _CONTENT_DELIVERY:
        if not form.isascii():
            if form in lower:
                return True
            continue
        if re.search(r"\b" + re.escape(form) + r"\b", lower):
            return True
    return False


# The direct OBJECT of a provision act that is an artifact: "give me a
# function" asks for the artifact to be BUILT, "give me an example of
# a function" asks for knowledge.  The object's class decides, never
# the verb.
_DELIVERY_ARTIFACT_OBJECT = re.compile(
    r"\b(?:give|gives|giving|gave|show|shows|showing|shown|tell|tells|"
    r"telling|told|send|sends|sending|sent|share|shares|sharing|"
    r"shared|describe|describes|describing|described|explain|"
    r"explains|explaining|explained|demonstrate|illustrate|present|"
    r"offer)(?:\s+\w+){0,3}\s+(?:a|an|the|some|this|that|my|our|"
    r"one)?\s*(?:"
    r"function|functions|class|classes|struct|enums?|interfaces?|"
    r"modules?|packages?|scripts?|programs?|code|apps?|applications?|"
    r"projects?|components?|snippets?|services?|apis?|endpoints?|"
    r"implementations?|코드|프로그램|함수|클래스|스크립트|앱|프로젝트)\b",
    re.IGNORECASE)


def _direct_artifact_object(clause: str) -> bool:
    """The provision act's object IS an artifact (it must be built)."""
    return bool(_DELIVERY_ARTIFACT_OBJECT.search(clause or ""))


# ── The PROCEDURE frame ──────────────────────────────────────────────
# The heads that denote an ordered way of doing something.  A request
# whose focused complement heads one asks for a PROCEDURE — the head's
# own semantic class decides this, never the activity the complement
# names ("kimchi stew", "fried rice", "the compiler" are all just the
# complement of the same structure).
_PROCEDURE_HEADS = frozenset({
    "procedure", "procedures", "steps", "step", "method", "methods",
    "way", "ways", "instructions", "instruction", "directions",
    "guide", "guides", "recipe", "recipes", "process", "routine",
    "workflow", "approach", "plan", "protocol", "recipe",
    "방법", "방식", "절차", "단계", "순서", "레시피", "요리법",
})

# The complement marker that follows a procedure head: the head heads
# the ACTIVITY it is a procedure for ("the way TO cook…", "the
# procedure FOR installing…", "the steps I should FOLLOW to sort…").
# The complement is the topic.  Up to five words may stand between the
# head and its marker — they carry the request's own framing ("steps
# should I follow to") — but the marker itself is required, so a
# predicate ("the steps are simple") never reads as a procedure.
_PROCEDURE_COMPLEMENT = re.compile(
    r"\b(?:procedure|procedures|steps?|methods?|ways?|instructions?|"
    r"directions?|guides?|recipes?|process|routines?|workflows?|"
    r"approach(?:es)?|plans?|protocols?)\b"
    r"(?:\s+\w+){0,5}?\s+(?:to|for|of|in|follow)\s+"
    r"([^,.;:?!]{2,90})",
    re.IGNORECASE)

# The "how" frame: the utterance's own predication of a METHOD, with
# its auxiliaries and subject stripped — what remains IS the activity.
_HOW_FRAME = re.compile(
    r"^\s*how\s+(?:(?:do|does|did|can|could|should|would|will|may|"
    r"might|is|are|was|were)\s+)?"
    r"(?:(?:i|we|you|he|she|they|it|one)\s+)?"
    r"(?:(?:do|does|did|to)\s+)?",
    re.IGNORECASE)

# A predication, not an activity: the how clause PREDICATES a property
# or a state ("how is X different", "how long is…"), so nothing in it
# can be done.
_HOW_PREDICATIVE = re.compile(
    r"^(that|this|those|these|it|they|he|she|there|is|are|was|were|"
    r"different|kinds?|types?|many|much|more|less|long|far|old|good|"
    r"better|easy|hard|possible|does|do|did)\b",
    re.IGNORECASE)


def _how_activity(clause: str) -> str:
    """The activity a how-question asks about, or "".

    "How can I configure the compiler?" asks about configuring the
    compiler: the frame ("how … I") is the request's predication of a
    method, the remainder is the activity.  A predication of a property
    and a clause carrying a reference are NOT activities — the former
    has no action in it, the latter must resolve first."""
    m = _HOW_FRAME.match(clause or "")
    if not m:
        return ""
    rest = clause[m.end():].strip(" ??.！～~")
    if not rest or _only_function_words(rest):
        return ""
    if _HOW_PREDICATIVE.match(rest):
        return ""
    if _anaphor_hits(clause.lower()):
        return ""
    return rest


# The heads that FRAME a knowledge request instead of naming its topic:
# "the latest news ABOUT X" is about X, not about news.  The head
# says WHAT KIND of knowledge is asked for; its complement says what it
# is about, and only the complement is a topic.
_INFORMATION_HEAD = re.compile(
    r"\b(?:news|information|info|details?|updates?|release|releases|"
    r"versions?|status|announcements?|progress|developments?|story|"
    r"stories|facts?|background|history|overview|situation|state|"
    r"latest|recent(?:ly)?|current(?:ly)?)\b"
    r"(?:\s+\w+){0,3}?\s+(?:about|on|regarding|concerning|for|from|in|"
    r"of|between)\s+([^,.;:?!]{2,90})",
    re.IGNORECASE)


def _information_complement(clause: str) -> str:
    """The topic a framed knowledge request is about, or "".

    "What's the latest news about black holes?" is about black holes:
    the frame ("latest news about") carries the request's temporal and
    categorical framing, never the topic."""
    m = _INFORMATION_HEAD.search(clause or "")
    if not m:
        return ""
    focus = m.group(1).strip(" ??.！～~")
    # strip the frame's own leftovers ("about the latest X")
    focus = re.sub(
        r"^(?:the\s+)?(?:latest|recent|current|newest)\s+",
        "", focus, flags=re.IGNORECASE).strip()
    return focus


def _procedure_complement(clause: str) -> str:
    """The activity a procedure-headed complement asks for, or "".

    The extraction reads the HEAD's complement — the grammatical shape
    of a procedure request — so any activity phrases the same way is
    recognized the same way.
    """
    m = _PROCEDURE_COMPLEMENT.search(clause or "")
    if not m:
        return ""
    activity = m.group(1).strip().rstrip(".?!~")
    # an infinitive marker left over from the filler ("the steps I
    # should follow TO sort…"): the activity is the verb phrase after it
    return re.sub(r"^to\s+", "", activity, flags=re.IGNORECASE).strip()


def _is_procedure_request(act: SemanticAct, request: str) -> bool:
    """Whether the utterance asks for an ordered way of doing something.

    Two structural sources, both about the REQUEST's shape:
      - a how-to clause (the interrogative/nominal "how" predication),
      - a focused complement headed by the procedure frame.
    An explanation, a comparison or a negated/past predication is
    answered as what it is — never re-read as a procedure.
    """
    if act is None:
        return False
    if act.negated or act.past or act.social_act:
        return False
    if act.explanation_request or act.predicate_class == "comparison":
        return False
    if act.method_question:
        return True
    return bool(_procedure_complement(request or ""))


def _is_request(clause: str) -> bool:
    """Whether the clause ASKS for something: an interrogative or an
    imperative — a report neither asks nor inherits."""
    c = (clause or "").strip()
    if not c:
        return False
    if c.endswith("?") or "？" in c:
        return True
    if re.match(r"^(what|who|whom|whose|which|how|why|when|where)\b",
                c, re.IGNORECASE):
        return True
    return _request_clause(c)


# ── The utterance's request units and their content focus ────────────
# An utterance often ASKS for several things at once: a report, an
# example, a constraint on the reply, a retry when the evidence is
# thin.  The UNITS come from the utterance's own coordination (a
# sentence boundary, a comma before a conjunction, a semicolon) and
# each unit's KIND from its predicate's grammatical class — knowledge
# work, investigation, a conditional instruction, a response-shaping
# demand.  Nothing here maps a sentence to an answer.

_UNIT_SPLIT = re.compile(
    r"\s*(?:,\s*(?:but|and|also|then|plus)\s+|\s+;\s+|\.\s+|"
    r"\s+but\s+|\s+then\s+)", re.IGNORECASE)

_COORD = re.compile(r"\s+(?:and|or)\s+", re.IGNORECASE)

# The predicates that perform KNOWLEDGE work over content: the
# request's predicate hands knowledge over, contrasts it or structures
# it.  The class is grammar — what the predicate DOES with knowledge —
# never a mapping from a sentence to an answer.
_KNOWLEDGE_ACT = re.compile(
    r"^(?:please\s+|also\s+|then\s+|just\s+|and\s+)?"
    r"(?:explain\w*|describ\w*|tell|tells|told|show|shows|shown|"
    r"give|gives|given|share\w*|send\w*|present\w*|offer\w*|"
    r"demonstrate\w*|illustrate\w*|define\w*|list|compare|contrast|"
    r"summar\w*|analy\w*|discuss\w*|clarif\w*|cover|enumerat\w*|"
    r"teach|walk\s+me\s+through|break\s+down|include|"
    r"알려|보여|설명|비교|정리|요약|나열|가르쳐)",
    re.IGNORECASE)

# The INVESTIGATION predicates: the addressee is asked to FIND OUT —
# the knowledge is not in the conversation yet.  Whether the act IS an
# investigation is decided by the OBJECT too: an artifact nominal
# ("review my code") is work on an artifact, never a research ask.
_INQUIRY_ACT = re.compile(
    r"^(?:please\s+|also\s+|then\s+|then\s+please\s+)?"
    r"(?:(?:can|could|would|will)\s+(?:you\s+)?|would\s+you\s+mind\s+"
    r"(?:researching|investigating|looking\s+into)\s+)?"
    r"(?:research\w*|investigate\w*|explore\w*|review|reviews|reviewing|"
    r"examine\w*|survey\w*|study|studying|look\s+into|dig\s+into|"
    r"find\s+out\s+about|조사|탐구|검토|살펴)",
    re.IGNORECASE)

# The artifact nominal: a request whose object NAMES an artifact asks
# for the artifact, never for an investigation of it.
_ARTIFACT_NOMINAL = re.compile(
    r"\b(?:function|functions|class|classes|struct|enums?|interfaces?|"
    r"modules?|packages?|scripts?|programs?|code|apps?|applications?|"
    r"projects?|components?|snippets?|services?|apis?|endpoints?|"
    r"implementations?|bug|bugs|repo|repos|repository|"
    r"코드|프로그램|함수|클래스|스크립트|앱|프로젝트|오류)\b",
    re.IGNORECASE)

# The response-shaping predicates: the unit constrains THE REPLY, it
# does not name content.  The predicate's own class decides this.
_RESPONSE_MANAGEMENT = re.compile(
    r"^(?:please\s+|also\s+|just\s+|and\s+)?"
    r"(?:keep|keeps|limit|limits|avoid|avoiding|restrict\w*|"
    r"make\s+sure|try|trying|aim|aiming|stay|remain|ensure|hold|"
    r"cap|capped|shorten|trim|do\s+not|don'?t|되게|말고|제한|지켜)",
    re.IGNORECASE)

# A measure phrase: an extent the reply must fit inside — a bound on
# the RESPONSE, never a topic of the discussion.
_MEASURE_PHRASE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|"
    r"thirty|forty|fifty|hundred|\d+)\s*"
    r"(?:minutes?|seconds?|hours?|days?|weeks?|words?|sentences?|"
    r"lines?|paragraphs?|pages?|characters?|steps?|items?|bullets?)\b",
    re.IGNORECASE)

# A conditional instruction: the unit states WHEN the previous act has
# to be repeated — the workflow's own shape, not a topic.
_CONDITIONAL = re.compile(
    r"\b(?:if|unless|should\s+(?:the|it|they)|in\s+case|given\s+that|"
    r"만약|경우에)\b", re.IGNORECASE)

_CONDITION_CLAUSE = re.compile(
    r"\b(?:if|unless|in\s+case|given\s+that)\s+(.+)$",
    re.IGNORECASE)

# The performative frame INSIDE a unit: the ask's own verb and its
# addressee are grammar — the complement is the content.
_UNIT_PERFORMATIVE = re.compile(
    r"^(?:please\s+|plz\s+|pls\s+|let'?s\s+)?"
    r"(?:(?:(?:can|could|would|will)\s+(?:you\s+)?\s*|"
    r"would\s+you\s+mind\s+)?"
    r"(?:research\w*|investigate\w*|explor\w*|examine\w*|survey|"
    r"stud(?:y|ies)|look\s+into|dig\s+into|find\s+out)"
    r"|(?:explain|describ\w*|tell\s+me|give\s+me|show\s+me|send\s+me|"
    r"offer\s+me|walk\s+me\s+through|talk\s+about|discuss|include|"
    r"cover|address|make|create|"
    r"build|write|generate|implement|develop|design|fix|repair|debug|"
    r"refactor|locate|list|print|calculate|remove|delete|discard|"
    r"compare|contrast|check|verify|determine|detect|identify|sort|"
    r"filter|select|return|store|save|read|process|"
    r"만들|생성|작성|구현|설명|찾|정렬|제거|삭제|출력|계산))\s+",
    re.IGNORECASE)

# The extent adverbial: how MUCH of the topic the reply must carry.
# It specifies the RESPONSE, so it never becomes part of the topic.
_EXTENT_ADVERBIAL = re.compile(
    r"(?:,?\s*\b(?:in\s+(?:more\s+)?detail|in\s+depth|in\s+full|"
    r"at\s+length|thoroughly|comprehensively|fully|briefly|concisely|"
    r"shortly|quickly|simply|plainly|easily|carefully|extensively|"
    r"in\s+simple\s+terms|in\s+plain\s+terms|step\s+by\s+step|"
    r"step-by-step|from\s+start\s+to\s+finish|end\s+to\s+end)\b|"
    r",?\s*(?:자세하게|상세하게|상세히|간단히|간결하게|단계별로|"
    r"구체적으로|쉽게|짧게))\s*$",
    re.IGNORECASE)

# The light predicate a how-question hosts: "how X WORKS" is about X —
# the light verb carries the question's form, never its content.
_HOW_PREDICATE_TAIL = re.compile(
    r"\s+(?:works?|happens?|behave[s]?|operate[s]?|function[s]?|"
    r"differ[s]?|change[s]?|looks?|seems?|grows?|develops?|"
    r"proceeds?|continues?|matter[s]?|succeed[s]?|is|are|was|were)\s*$",
    re.IGNORECASE)

_WH_HEAD = re.compile(r"^(how|what|why|when|where|who|which|whether)\b",
                      re.IGNORECASE)


def _split_units(text: str) -> list[dict]:
    """The utterance's request units WITH their spans, from its own
    coordination.  Each unit remembers where it stands in the raw text
    so the segmentation stays traceable to what was written."""
    source = text or ""
    pieces: list[tuple[int, int]] = []
    cursor = 0
    for m in _UNIT_SPLIT.finditer(source):
        if m.start() > cursor:
            pieces.append((cursor, m.start()))
        cursor = m.end()
    if cursor < len(source):
        pieces.append((cursor, len(source)))
    units: list[dict] = []
    for start, end in pieces:
        units.extend(_split_coordinated(source, start, end))
    return [u for u in units if u["text"].strip()]


def _split_coordinated(source: str, start: int, end: int) -> list[dict]:
    """A coordination splits only when the tail PERFORMS: "X and tell
    me Y" is two asks, "salt and pepper" is one noun phrase."""
    text = source[start:end]
    cuts: list[tuple[int, int]] = []
    cursor = 0
    for m in _COORD.finditer(text):
        if m.start() > cursor:
            cuts.append((cursor, m.start()))
        cursor = m.end()
    if cursor < len(text):
        cuts.append((cursor, len(text)))
    if len(cuts) < 2:
        return [{"start": start, "end": end, "text": text}]
    merged: list[list] = []
    for cut_start, cut_end in cuts:
        tail = text[cut_start:cut_end]
        absolute = (start + cut_start, start + cut_end)
        if _KNOWLEDGE_ACT.match(tail.strip()) or \
                _INQUIRY_ACT.match(tail.strip()):
            merged.append([absolute[0], absolute[1], tail])
        elif merged:
            merged[-1][1] = absolute[1]
            merged[-1][2] = source[merged[-1][0]:merged[-1][1]]
        else:
            merged.append([absolute[0], absolute[1], tail])
    return [{"start": s, "end": e, "text": t} for s, e, t in merged]


def _content_focus(unit: str) -> tuple[str, str]:
    """The unit's content focus and the ask's aspect, or ("", aspect).

    The frame is stripped first (the ask's own verb and addressee are
    grammar), then the complement is read by its SHAPE: an embedded
    interrogative ("explain HOW X works"), a framed knowledge request
    ("the history of X"), a procedure complement, an about-complement,
    or the complement itself.  The extent adverbial specifies the
    REPLY, so it never enters the focus."""
    focus = _UNIT_PERFORMATIVE.sub("", unit, count=1).strip()
    if not focus:
        return "", ""
    aspect = ""
    head = _WH_HEAD.match(focus)
    if head:
        aspect = focus
        focus = _QUESTION_FRAMING.sub("", focus).strip()
        if head.group(1).lower() == "how":
            focus = _HOW_PREDICATE_TAIL.sub("", focus).strip()
    else:
        found = ""
        for extractor in (_information_complement, _procedure_complement,
                          _how_activity):
            found = extractor(focus) or ""
            if found:
                break
        if found:
            focus = found
        else:
            m = re.search(r"\b(?:about|regarding|concerning)\s+"
                          r"([^,.;:?!]{2,90})", focus, re.IGNORECASE)
            if m:
                focus = m.group(1)
    for _ in range(3):
        trimmed = _EXTENT_ADVERBIAL.sub("", focus).strip(" \t,.;:!?")
        if trimmed == focus or not trimmed:
            break
        focus = trimmed
    if aspect and _WH_HEAD.match(aspect) and \
            _WH_HEAD.match(aspect).group(1).lower() == "how":
        focus = _HOW_PREDICATE_TAIL.sub("", focus).strip()
    focus = re.sub(r"^(a|an|the)\s+", "", focus,
                   flags=re.IGNORECASE).strip()
    focus = focus.rstrip(" \t.,;:!?").strip()
    if not focus or len(focus) > 100 or _only_function_words(focus):
        return "", aspect
    return focus, aspect


def _focus_topics(focus: str) -> list[str]:
    """The focus's own members: a contrast names both of them."""
    parts = [p.strip(" \t.") for p in re.split(
        r"\s+(?:and|versus|vs\.?|와|과|이랑|랑)\s+", focus or "",
        flags=re.IGNORECASE)]
    if len(parts) >= 2 and all(len(p) >= 2 for p in parts[:2]):
        return _dedupe(parts[:2])
    return _dedupe([focus]) if focus else []


def _has_anaphora(text: str) -> bool:
    lower = (text or "").lower()
    return bool(_anaphor_hits(lower)) or \
        any(p in lower for p in _TOPIC_ANAPHORA_PHRASES)


@dataclass
class UtteranceFocus:
    """The utterance's CONTENT structure: the main topic, the
    information goals it carries as children, the constraints the
    reply must respect and the workflow the request spells out."""
    topics: list[str] = field(default_factory=list)
    aspect: str = ""
    subs: list[dict] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    units: list[dict] = field(default_factory=list)
    inquiry: str = ""
    workflow: list[dict] = field(default_factory=list)
    knowledge_act: bool = False
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "topics": list(self.topics),
            "aspect": self.aspect,
            "subs": [dict(s) for s in self.subs],
            "constraints": list(self.constraints),
            "inquiry": self.inquiry,
            "workflow": [dict(w) for w in self.workflow],
            "knowledge_act": self.knowledge_act,
            "source": self.source,
        }


def analyze_focus(text: str, discourse_topic: str = "",
                  previous_turns: list | None = None) -> UtteranceFocus:
    """Read the utterance's content structure from its own grammar.

    Units → each unit's KIND (knowledge work / investigation /
    conditional instruction / response-shaping demand / report) → each
    knowledge unit's focus → the main topic, with the remaining
    knowledge units as CHILD information goals.  An anaphor inside a
    child resolves against the utterance's own main topic first (the
    child's parent supplies the referent), then against the discourse."""
    raw = (text or "").strip()
    if not raw:
        return UtteranceFocus()
    out = UtteranceFocus()
    knowledge: list[dict] = []
    for unit in _split_units(raw):
        text_unit = unit["text"].strip()
        if not text_unit:
            continue
        if _INQUIRY_ACT.match(text_unit) and \
                not _ARTIFACT_NOMINAL.search(text_unit):
            focus, aspect = _content_focus(text_unit)
            kind = "inquiry"
        elif _KNOWLEDGE_ACT.match(text_unit):
            focus, aspect = _content_focus(text_unit)
            kind = "knowledge"
        elif _CONDITIONAL.search(text_unit):
            focus, aspect = "", ""
            kind = "conditional"
        elif _RESPONSE_MANAGEMENT.match(text_unit) or \
                _MEASURE_PHRASE.search(text_unit):
            focus, aspect = "", ""
            kind = "constraint"
        else:
            focus, aspect = "", ""
            kind = "report"
        out.units.append({"text": text_unit, "kind": kind,
                          "focus": focus, "span": [unit["start"],
                                                   unit["end"]]})
        if kind == "inquiry" and focus:
            out.inquiry = focus
        elif kind == "knowledge" and focus:
            knowledge.append({"text": text_unit, "focus": focus,
                              "aspect": aspect,
                              "span": [unit["start"], unit["end"]]})
            out.knowledge_act = True
        elif kind == "constraint":
            out.constraints.append(text_unit)
        elif kind == "conditional":
            m = _CONDITION_CLAUSE.search(text_unit)
            out.workflow.append({
                "action": "reformulate_and_search_again",
                "condition": (m.group(1).strip(" \t.!?")
                              if m else text_unit),
            })
    if out.inquiry:
        out.knowledge_act = True
        out.workflow.insert(0, {"action": "search", "condition": ""})
    # The main topic: the investigation's subject when the utterance
    # INVESTIGATES, else the first knowledge focus that is not itself
    # a reference — a child resolves against it, never the reverse.
    main_index = -1
    if out.inquiry:
        out.topics = _focus_topics(out.inquiry)
    else:
        for index, item in enumerate(knowledge):
            if _has_anaphora(item["focus"]):
                continue
            out.topics = _focus_topics(item["focus"])
            out.aspect = item["aspect"]
            main_index = index
            break
    if out.topics:
        out.source = "utterance"
        parent = out.topics[0]
        for index, item in enumerate(knowledge):
            if index == main_index:
                continue
            focus = item["focus"]
            subject = parent
            if _has_anaphora(focus):
                if not out.topics:
                    continue
                focus = _bind_anaphor(focus, parent)
            else:
                subject = parent
            out.subs.append({
                "focus": focus,
                "parent": subject,
                "asks_for": item["aspect"] or "knowledge",
                "span": list(item.get("span") or []),
            })
    return out


def inquiry_request(request: str) -> str:
    """The investigation's subject when the utterance ASKS THE
    ADDRESSEE TO FIND OUT, else "".

    The predicate's own class decides this (an investigation, never a
    build) and the OBJECT guards it: an artifact nominal ("review my
    code") is work on an artifact, not a research ask."""
    if not request or not request.strip():
        return ""
    try:
        return analyze_focus(request).inquiry
    except Exception:
        return ""


def _bind_anaphor(focus: str, subject: str) -> str:
    """The child's reference bound to the utterance's own subject:
    'how it helps life on Earth' asked AFTER a topic is about THAT
    topic — the parent supplies the referent."""
    if not subject or not focus:
        return focus
    hits = _anaphor_hits(focus.lower())
    if not hits:
        return focus
    word = hits[0][1]
    return re.sub(r"\b" + re.escape(word) + r"\b", subject, focus,
                  count=1, flags=re.IGNORECASE)


def _elliptical_topic(clause: str, discourse_topic: str) -> list[str]:
    """A content request that names no topic of its own: the referent
    is the DISCUSSION's.

    Only a request inherits — a system question ("What can you do?")
    and a report ("Hi") ask nothing of the discourse — and only the
    topic state supplies it, never the sentence's own framing.
    """
    if not discourse_topic or not discourse_topic.strip():
        return []
    if not _is_request(clause):
        return []
    if not _content_delivery_request(clause):
        return []
    return _dedupe([discourse_topic.strip()])


_PROPER_CONNECTOR = (r"(?:of|the|de|di|da|van|von|del|der|la|le|"
                     r"du|dos|das|al)")
_PROPER_PHRASE_RE = re.compile(
    r"(?<![A-Za-z0-9_+#])"
    r"([A-Z][A-Za-z0-9+#]{1,20}"
    r"(?:\s+" + _PROPER_CONNECTOR + r"(?:\s+" + _PROPER_CONNECTOR
    + r")*\s+[A-Z][A-Za-z0-9+#]{1,20}"
    r"|\s+[A-Z][A-Za-z0-9+#]{1,20})*"
    r"(?:\s+\d{1,4})?)")


def _restore_proper_phrases(topics: list[str], text: str) -> list[str]:
    """A topic that is a FRAGMENT of a proper name is restored to the
    full name the utterance states: "Americas" must not become the
    subject of a question about "Circuit of the Americas", and the
    lowercase spelling of a name ("circuit of the America") is still
    the name.  General structure, no word list of subjects: a
    capitalized run joined by name connectors is one name, and an
    entity modifier directly attached to a noun ("COTA 서킷") attaches
    to it."""
    if not topics or not text:
        return topics
    phrases = [m.group(1).strip()
               for m in _PROPER_PHRASE_RE.finditer(text)]
    out: list[str] = []
    for topic in topics:
        t = str(topic or "").strip()
        if not t:
            continue
        restored = ""
        for phrase in phrases:
            if len(phrase) > len(t) and len(phrase.split()) <= 6 and \
                    re.search(r"(?<![A-Za-z0-9])" + re.escape(t)
                              + r"(?![A-Za-z0-9])", phrase):
                restored = phrase
                break
        if restored:
            out.append(restored)
            continue
        pos = text.find(t)
        if pos > 0:
            left = text[:pos]
            m = re.search(
                r"(?:^|(?:about|on|in|for|of|regarding)\s+)"
                r"(\w+)\s+" + _PROPER_CONNECTOR
                + r"(?:\s+" + _PROPER_CONNECTOR + r")*\s*$",
                left, re.IGNORECASE)
            if m:
                out.append(text[m.start(1):pos + len(t)].strip())
                continue
            m = re.search(r"([A-Z][A-Za-z0-9+#]{1,20})\s+$", left)
            if m:
                out.append(text[m.start(1):pos + len(t)].strip())
                continue
        out.append(t)
    seen: set[str] = set()
    final: list[str] = []
    for t in out:
        key = t.lower()
        if key and key not in seen:
            seen.add(key)
            final.append(t)
    return final or topics


def extract_topics(text: str, semantic=None,
                   previous_turns: list | None = None,
                   discourse_topic: str = "",
                   source: dict | None = None) -> list[str]:
    """The utterance's content topics: the interrogative's focused
    constituent with the WH-framing removed, plus the antecedents an
    anaphoric reference points at.

    Generic anaphora resolution: a pronoun in the question refers to
    the salient noun phrases of the SAME utterance first, then to the
    conversation's resolved topic state, then to the previous turn.
    Never a sentence→answer map.

    Ellipsis: a REQUEST that names no content of its own takes the
    referent from the discourse's topic state ("Can you give me an
    example?" — an instance of what the conversation is about).  The
    request's framing supplies no topic; the topic state does.

    `source` (optional out-dict) records WHERE the referent came from —
    `utterance`, `discourse_topic`, `previous_turn` or `none` — so the
    reference resolution is observable, not silent.
    """
    def _finish(topics: list[str], kind: str) -> list[str]:
        topics = _restore_proper_phrases(list(topics or []), raw)
        if source is not None:
            source["reference_source"] = kind
        return topics

    raw = text.strip()
    # A leading performative marker is grammar: strip it from both the
    # salience pool and the clause.
    raw_stripped = _LEADING_PERFORMATIVE.sub("", raw, count=1).strip()
    if raw_stripped:
        raw = raw_stripped
    salient = _salient_nouns(raw)
    clause = _query_clause(raw)
    clause = _LEADING_PERFORMATIVE.sub("", clause, count=1).strip()
    clause_lower = clause.lower()
    hits = _anaphor_hits(clause_lower)
    has_anaphora = bool(hits) or \
        any(p in clause_lower for p in _TOPIC_ANAPHORA_PHRASES)

    topic_phrase = _QUESTION_FRAMING.sub("", clause).strip()
    topic_phrase = topic_phrase.rstrip("?？.!~").strip()

    # A comprehension gap's own embedded interrogative is the focused
    # content: the speaker names WHAT they fail to understand, so a
    # domain noun inside that complement is only a fallback.
    gap = _gap_complement(clause)
    if gap:
        topic_phrase = gap

    # The utterance's OWN content structure (units → focus → child
    # information goals).  It answers when the utterance performs
    # knowledge work: the ask's frame is grammar, its complement is the
    # topic, and a child reference resolves against the parent.  A gap
    # or a bare reference is left to the anaphora rules below — the
    # structure supplies no referent the discourse must.
    if not gap:
        focus = analyze_focus(text, discourse_topic, previous_turns)
        if focus.topics:
            # A focus topic that IS the utterance (the request echoed
            # back) names no topic: the request's own framing is
            # grammar, and echoing it would answer nothing.  Dropping
            # it lets the clause's own content nouns be the topic.
            body = raw.strip().rstrip("??.!~").strip().lower()
            out = [t for t in _dedupe(list(focus.topics))
                   if t.strip().rstrip("??.!~").strip().lower() != body
                   and len(t.strip()) < max(40, len(body) - 12)]
            if out and all(len(t) <= 100 for t in out):
                return _finish(out, focus.source or "utterance")

    # A real content noun (a language, a framework, a domain) is the
    # topic whenever the utterance names one — unless the utterance
    # itself states the proposition it is about.
    if salient and not has_anaphora and not gap:
        if len(salient) >= 2:
            # A comparative structure over two named topics
            if re.search(r"\b(vs\.?|versus|and|와|과|이랑|랑)\b", clause) \
                    or "vs" in clause_lower:
                return _finish(_dedupe(salient[:2]),
                               "utterance")
        return _finish(_dedupe(salient[:1]), "utterance")

    # The activity the request is focused on IS the topic, never the
    # request's own framing.  Two structural sources: a procedure head
    # heads its complement ("the procedure FOR installing…") — which
    # wins even over a reference inside it, because the head already
    # marks the complement as the focused content — and the how
    # frame's remainder ("How can I configure…?"), which yields to a
    # reference so the reference can resolve first.
    if not gap:
        activity = _procedure_complement(clause)
        if not activity and not has_anaphora:
            activity = _how_activity(clause)
        if activity and not _only_function_words(activity) \
                and len(activity) <= 80:
            return _finish([_clean_phrase(activity)], "utterance")

    # A framed knowledge request: the frame says WHAT KIND of knowledge
    # is asked for ("the latest news about"), the complement says what
    # it is ABOUT — only the complement is a topic.
    if not gap and not has_anaphora:
        focus = _information_complement(clause)
        if focus and not _only_function_words(focus) \
                and len(focus) <= 80:
            return _finish([_clean_phrase(focus)], "utterance")

    if has_anaphora:
        # An antecedent PRECEDES its anaphor: a same-utterance salient
        # noun after the pronoun is not an antecedent ("explain that in
        # a simpler way" — "way" follows "that"), so with nothing
        # preceding the reference resolves from the discourse state,
        # then from the previous turn.
        ana_pos = hits[0][0] if hits else -1
        preceding = [s for s in salient if ana_pos >= 0 and
                     0 <= clause_lower.find(s.lower()) < ana_pos]
        if len(preceding) >= 2:
            return _finish(_dedupe(preceding[:2]), "utterance")
        if len(preceding) == 1:
            return _finish(_dedupe(list(preceding)), "utterance")
        if discourse_topic and discourse_topic.strip():
            return _finish(_dedupe([discourse_topic.strip()]),
                           "discourse_topic")
        if previous_turns:
            for turn in reversed(list(previous_turns)[-3:]):
                prev = turn if isinstance(turn, str) else \
                    (turn.get("user_input", "") if isinstance(turn, dict)
                     else getattr(turn, "user_input", ""))
                prev_salient = _salient_nouns(prev)
                if prev_salient:
                    return _finish(_dedupe(prev_salient[:2]),
                                   "previous_turn")
        # A reference with NOTHING to refer to: the utterance carries
        # no topic of its own, and no discourse supplies one.  The
        # reference is what is missing, so NO topic is stated — the
        # whole sentence is never the topic (echoing it would answer
        # nothing), and the intent layer asks for the referent.
        return _finish([], "none")

    if not topic_phrase:
        inherited = _elliptical_topic(clause, discourse_topic)
        return _finish(inherited,
                       "discourse_topic" if inherited else "none")
    # A comparative structure: "X and Y" / "X vs Y"
    parts = re.split(r"\s+(?:and|versus|vs\.?|와|과|이랑|랑)\s+",
                     topic_phrase, flags=re.IGNORECASE)
    if len(parts) == 2 and all(len(p.strip()) >= 2 for p in parts):
        return _finish(_dedupe([p.strip() for p in parts]),
                       "utterance")

    cut = re.sub(r"^(a|an|the)\s+", "", topic_phrase,
                 flags=re.IGNORECASE).strip()
    cut = _TOPIC_CUT.sub("", cut).strip() or cut
    if not cut or len(cut) > 80 or _only_function_words(cut):
        inherited = _elliptical_topic(clause, discourse_topic)
        return _finish(inherited,
                       "discourse_topic" if inherited else "none")
    return _finish([cut], "utterance")


def _clean_phrase(phrase: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", phrase,
                  flags=re.IGNORECASE).strip()


def _only_function_words(phrase: str) -> bool:
    """The phrase names nothing: every word of it is grammar — a closed
    class, a filler, or the request's own framing.

    The check is over the GRAMMATICAL words, never over _NON_TOPIC:
    that set also excludes common knowledge nouns ("function",
    "language", "code") from the proper-noun SALIENCE pool, and a
    salience opinion is not a judgement about whether a word carries
    content — "What is a function?" is about a function.  Korean words
    are judged the same way, against their own closed class after the
    particle is stripped."""
    words = [w.strip(".,!?…~\"'") for w in phrase.split()]
    return bool(words) and all(_is_grammatical_word(w) for w in words)


def _is_grammatical_word(word: str) -> bool:
    """Whether the word is grammar in its own language."""
    w = (word or "").strip(".,!?…~\"'").lower()
    if not w:
        return True
    if w in _GRAMMAR_WORDS:
        return True
    if _HAS_HANGUL.search(w):
        if _ko_non_topic(w) or _ko_strip_particle(w) in _KO_NON_TOPIC:
            return True
    return False


# The grammatical words: closed classes, fillers and the request's own
# framing.  A phrase made only of these names nothing.
_GRAMMAR_WORDS = frozenset({
    # pronouns, determiners, articles
    "i", "we", "you", "they", "it", "my", "your", "our", "the", "a",
    "an", "this", "that", "these", "those", "me", "us", "him", "her",
    "them", "his", "its", "one",
    # wh-words and the framing nouns they head
    "what", "how", "why", "when", "where", "which", "who", "whom",
    "whose", "whether", "kind", "type",
    # auxiliaries and copulas
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "would", "should", "will", "am", "have", "has", "had", "been",
    "need", "want", "like",
    # conjunctions, prepositions, fillers, quantifiers
    "and", "or", "if", "so", "but", "then", "now", "also", "just",
    "really", "very", "much", "many", "few", "some", "any", "all",
    "given", "longest", "first", "second", "third", "whether",
    # greetings and politeness
    "hi", "hello", "hey", "thanks", "thank", "please", "yes", "no",
    "ok", "okay", "sure",
    # the request's own performative heads: grammar when a phrase is
    # nothing but the ask itself
    "make", "create", "build", "write", "begin", "start", "focus",
    "hear", "heard", "think", "thinking", "experience", "beginner",
    "realistic", "explain", "describe", "remove", "delete", "compare",
    "contrast", "check", "verify", "determine", "detect", "identify",
    "apply", "transform", "convert", "merge", "split", "keep", "select",
    "return", "read", "store", "save", "process", "calculate", "print",
    "show", "locate", "tell", "ask", "give", "help", "find", "sort",
    "descending", "ascending", "order",
})



def _dedupe(items: list[str]) -> list[str]:
    out, seen = [], set()
    for i in items:
        k = i.strip().lower()
        if i.strip() and k not in seen:
            seen.add(k)
            out.append(i.strip())
    return out


# ----------------------------------------------------------------------
# Intent derivation
# ----------------------------------------------------------------------

# The KNOWLEDGE performative: the turn opens by asking to be TOLD.
# It is a MOOD lexicon (the same grammar as the imperative markers
# above), never a topic word: what follows it is content, and an
# artifact goal would restate the request as its opposite — unless the
# turn's object IS an artifact ("explain how to write a function"
# names the function it wants).
_KNOWLEDGE_PERFORMATIVE = re.compile(
    r"^\s*(?:please\s+|just\s+|also\s+|could\s+you\s+|can\s+you\s+|"
    r"would\s+you\s+|let\s+(?:me\s+)?|help\s+me\s+(?:to\s+)?|"
    r"want\s+to\s+learn\s+)?"
    r"(teach|explain|describe|show|tell|define|summar\w*|illustrate|"
    r"discuss|cover|clarify|walk\s+(?:me\s+)?through|compare|"
    r"contrast|list|enumerate|demonstrate|review|instruct|train|"
    r"tutor|coach|guid\w*|inform\w*|"
    r"알려|설명|가르쳐|보여|정의|비교|요약|나열|목록화|알려줘|가르쳐줘)",
    re.IGNORECASE)


def _asks_to_be_told(request: str) -> bool:
    """Whether the utterance OPENS with a knowledge performative: it
    asks to be TOLD something.  A later sentence that reports the
    speaker's position ("I have never written code before") is
    background — it never turns a "teach me" into a request to build."""
    return bool(_KNOWLEDGE_PERFORMATIVE.match(str(request or "").strip()))


def _action_unit(focus) -> bool:
    """Whether the utterance carries a unit that PREDICATES AN ACTION
    rather than asking for knowledge, an investigation, a constraint or
    a condition — the only kind of unit an artifact request can come
    from."""
    for unit in getattr(focus, "units", None) or []:
        if str(unit.get("kind") or "") == "report":
            return True
    return False


def asks_for_content(act: SemanticAct, request: str = "") -> tuple[bool, str]:
    """Does this utterance ASK FOR CONTENT — knowledge to be told, an
    answer to be given, an explanation to be read — rather than for an
    artifact?

    This is what distinguishes a turn the parser may still GUESS an
    artifact goal for ("I don't know where to begin" — the guess is
    the parser's, and the route never follows it) from a turn whose
    own wording already names the content it wants ("Teach me
    programming", "I want a guide to X"): the second may never be
    promoted to an artifact goal, because that would restate the
    request as its opposite.

    Returns (content_asked, evidence).
    """
    if not request:
        return False, ""
    if act.form == FORM_INTERROGATIVE or act.force == FORCE_QUERY:
        return True, "query act: the addressee is asked a question, " \
                     "never for an artifact"
    focus = analyze_focus(request)
    if _asks_to_be_told(request) and not _direct_artifact_object(request):
        return True, ("the turn opens with a knowledge performative: "
                      "content is asked for, never an artifact")
    if focus.inquiry and not _direct_artifact_object(request):
        return True, ("inquiry act: the addressee is asked to find "
                      "out, not to build")
    if act.subject == "first" and act.desiderative and \
            act.predicate_class == "action":
        if not _direct_artifact_object(request):
            return True, ("first-person desire whose object is not an "
                          "artifact: content is asked for")
    return False, ""


def allows_artifact_creation(act: SemanticAct,
                             request: str = "") -> tuple[bool, str]:
    """May this utterance's semantic content be bound to a CREATE goal?

    The rule: an artifact is created only for a REQUEST act (the speaker
    asks the addressee to act) or for an action predication that is not
    a query, a social remark, a comparison, a knowledge statement, a
    first-person report or a desire.  A mere MENTION of a topic — however
    knowledge-rich — never creates an artifact goal.  Two predicates can
    never build anything: an INVESTIGATION (the addressee is asked to
    find out) and a CONTENT-DELIVERY act whose object is not itself an
    artifact ("tell me the key milestones" asks for knowledge, never for
    a program).

    Returns (allowed, evidence).
    """
    if act.form == FORM_INTERROGATIVE or act.force == FORCE_QUERY:
        return False, "query act: no creation is requested"
    if act.force == FORCE_EXPRESS:
        return False, "social act: no creation is requested"
    if act.negated or act.past:
        return False, "negated/past predication: no creation requested"
    if request:
        focus = analyze_focus(request)
        if focus.inquiry and not _direct_artifact_object(request):
            return False, ("inquiry act: the addressee is asked to "
                           "find out, not to build")
        if focus.knowledge_act and not _action_unit(focus):
            return False, ("knowledge-work act: the addressee is asked "
                           "for content, never for an artifact")
        if _asks_to_be_told(request) and \
                not _direct_artifact_object(request):
            return False, ("the turn opens with a knowledge "
                           "performative: content is asked for, "
                           "never an artifact")
        if _is_procedure_request(act, request) and \
                not _direct_artifact_object(request):
            return False, ("procedure request: the addressee is asked "
                           "for an ordered way, never for an artifact")
    if act.force == FORCE_REQUEST:
        return True, "request act: the addressee is asked to act"
    if act.predicate_class in ("social", "state", "comparison",
                               "knowledge", "advice", "evaluation"):
        return False, f"predicate={act.predicate_class}: not an action"
    if act.subject == "first":
        # An indirect request: a first-person desire that predicates an
        # ACTION ("I want to build an app") still asks for the act —
        # but only when what is desired IS an artifact.  A desire whose
        # object is knowledge ("I want a guide to learning X") asks for
        # CONTENT: an artifact goal may never claim it.
        if act.desiderative and act.predicate_class == "action":
            if not request:
                return True, "first-person desire predicated as an action"
            if _direct_artifact_object(request):
                return True, "first-person desire over an artifact object"
            return False, ("first-person desire whose object is not an "
                           "artifact: content is asked for, never code")
        return False, ("first-person report: "
                       "no creation is requested")
    if not act.predicate_class:
        return False, "no action predicate: no creation is requested"
    return True, "action predicate in a non-query act"


def derive_intent(semantic, act: SemanticAct, request: str,
                  knowledge: dict | None = None,
                  previous_turns: list | None = None) -> DialogueIntent:
    """Derive the intent type from the ACT (force + predicate class) and
    the CONTENT's kind (what the semantic representation carries).

    The topic words never enter this decision.
    """
    knowledge = knowledge or {}
    goal = getattr(semantic, "goal", None)
    goal_type = getattr(goal, "goal_type", None)
    goal_value = goal_type.value if goal_type is not None else ""
    artifact = getattr(goal, "artifact", None) or {}
    existing_artifact = any(
        str(e).startswith(("parent context", "explicit existing",
                           "artifact knowledge"))
        for e in (getattr(goal, "evidence", None) or []))

    reference: dict = {}
    topics = extract_topics(request, semantic, previous_turns,
                            str(knowledge.get("discourse_topic") or ""),
                            reference)
    intent = DialogueIntent(topics=topics)
    # The reference resolution travels with the intent: WHERE the
    # referent came from (this utterance, the conversation's topic
    # state, the previous turn) — so a reference is observable in the
    # understanding, never a silent substitution.
    if reference.get("reference_source"):
        intent.scope["reference_source"] = reference["reference_source"]
    _discourse = str(knowledge.get("discourse_topic") or "")
    if _discourse:
        intent.scope["discourse_topic"] = _discourse
    if previous_turns:
        intent.scope["previous_turns"] = len(list(previous_turns))

    shell_context = bool(knowledge.get("shell_context"))
    natural_language_topic = bool(knowledge.get("natural_language_topic"))
    self_topic = bool(knowledge.get("self_topic"))
    program_topic = bool(knowledge.get("program_topic"))
    # The semantic representation already carries an artifact goal (the
    # parser verified the structure): no provision reading may claim
    # this turn — the request asks for the thing, not for knowledge.
    _artifact_goal = goal_value in (GoalValue.project, GoalValue.function,
                                    GoalValue.klass, GoalValue.modify)

    def _set(t: IntentType, goal_text: str, conf: float, *ev: str):
        intent.type = t
        intent.goal = goal_text
        intent.confidence = conf
        intent.evidence.extend(ev)

    # ── 1. Social expressive act ──
    if act.force == FORCE_EXPRESS:
        if act.social_act == "salutation":
            _set(IntentType.GREETING, "exchange_greeting", 0.9,
                 "force=express", "predicate=social",
                 f"act={act.social_act}")
        else:
            _set(IntentType.CONVERSATION, f"social_{act.social_act or 'act'}",
                 0.85, "force=express", "predicate=social")
        return intent

    # ── 1b. Self-presentation: the addressee is asked to PRESENT
    # themselves ("introduce yourself", "자기소개해줘").  The act's
    # object is the ADDRESSEE, not a body of content, so no knowledge
    # retrieval and no explanation may claim the turn — it is an
    # interaction. ──
    if not (act.negated or act.past) \
            and act.force in (FORCE_REQUEST, FORCE_QUERY) \
            and _self_presentation_request(request):
        _set(IntentType.CONVERSATION, "present_self", 0.85,
             f"force={act.force}", "self-presentation performative")
        return intent

    # ── 2. Explanation request — a query act about a meaning ──
    if act.explanation_request and not (act.negated or act.past):
        # The register the request ASKS FOR refines the act within the
        # same class: an explanation asked for in a simpler way is a
        # SIMPLIFYING explanation of the same target.  The style
        # adverbial decides this — never the topic words.
        style = requested_style(request)
        if style == "simple":
            _set(IntentType.SIMPLIFY_EXPLANATION, "simplify_explanation",
                 0.8, "predicate=explanation performative",
                 "register=simple")
        else:
            _set(IntentType.EXPLANATION_REQUEST, "explain_target", 0.8,
                 "predicate=explanation performative")
        return intent

    # ── 2b. Procedure request — an ordered way of doing something ──
    # The REQUEST's shape decides this: a how-to clause, or a focused
    # complement headed by the procedure frame.  The activity the
    # complement names never enters the decision, so every request
    # phrased this way is understood the same way.  A shell
    # environment still answers its own how-to as the command query.
    if _is_procedure_request(act, request):
        if shell_context:
            _set(IntentType.COMMAND_QUERY, "how_to_in_shell", 0.8,
                 "method question" if act.method_question
                 else "procedure frame",
                 "shell context")
            intent.scope["platform"] = knowledge.get("platform", "")
            return intent
        _set(IntentType.PROCEDURAL_INSTRUCTION, "explain_how_to_proceed",
             0.8,
             "method question" if act.method_question
             else "procedure frame",
             "procedure request")
        return intent

    # ── 2c. Investigation request: the addressee is asked to FIND
    # OUT.  The predicate's own class decides this — never the topic
    # words.  The knowledge is not in the conversation yet, so the
    # turn's meaning is an investigation (search → evaluate →
    # reformulate → search → synthesize), never an explanation of what
    # is already known and never an artifact request.  A request whose
    # OBJECT is an artifact ("review my code") is work on an artifact
    # and keeps its own goal. ──
    investigation = inquiry_request(request)
    if investigation and not (act.negated or act.past) \
            and not _direct_artifact_object(request):
        _set(IntentType.RESEARCH, "investigate_target", 0.8,
             f"force={act.force}", "investigation predicate")
        intent.scope["investigation_subject"] = investigation
        # The workflow the REQUEST spelled out (a conditional
        # instruction's shape), else the investigation's own stages —
        # recorded so the plan can show how the evidence was sought.
        _focus = analyze_focus(request)
        intent.scope["workflow"] = list(_focus.workflow) or [
            {"action": "search", "condition": ""},
            {"action": "evaluate", "condition": ""},
            {"action": "reformulate_and_search_again",
             "condition": "the first results are thin"},
            {"action": "synthesize", "condition": ""},
        ]
        return intent

    # ── 2d. Knowledge work: the utterance OPENS with a knowledge
    # performative ("Teach me…", "Explain how…") and its focus carries
    # knowledge work.  The turn asks for content — never an
    # interaction, never an artifact — whatever sentence form carries
    # it: a declarative "Teach me X…" is a content request, not a
    # remark and not a program. ──
    if not (act.negated or act.past):
        _focus_act = analyze_focus(request)
        if _focus_act.knowledge_act and _asks_to_be_told(request) \
                and not _content_delivery_request(request):
            _set(IntentType.EXPLANATION_REQUEST, "explain_target", 0.78,
                 "knowledge performative",
                 "content requested, not interaction")
            return intent

    # ── 2f. Stance request: the act asks whether the addressee AGREES
    # with a proposition the utterance ITSELF carries.  What the turn
    # wants is a position on that claim, so the reply discusses the
    # claim from evidence — never a definition of the topic it
    # mentions, never a social acknowledgment. ──
    if not act.past and _stance_request(request) \
            and _carries_proposition(request):
        _set(IntentType.OPINION_REQUEST, "take_a_stance", 0.8,
             "interrogative predicates the addressee's agreement",
             "the utterance carries its own proposition")
        return intent

    # ── 2g. Proposal request: the act's frame DELIVERS content but
    # its complement is an INDEFINITE candidate with no topic to
    # retrieve — the addressee must produce the option.  The
    # complement's own shape decides this; the activity it would be
    # about never does. ──
    if not (act.negated or act.past) and act.form != FORM_INTERROGATIVE \
            and not _artifact_goal and _proposal_candidate(request) \
            and (_content_delivery_request(request)
                 or act.predicate_class == "advice"):
        _set(IntentType.PROPOSAL_REQUEST, "propose_candidate", 0.78,
             "content delivery act", "indefinite candidate complement")
        return intent

    # ── 2c. Provision request: the addressee is asked to PROVIDE
    # knowledge — never to build an artifact.  The performative head
    # asks for CONTENT ("tell me about X", "X에 대해 알려줘"), so the
    # turn's knowledge is retrieved and composed, whichever clause
    # carries the request. ──
    if act.force == FORCE_REQUEST and not (act.negated or act.past) \
            and not _artifact_goal and _content_delivery_request(request) \
            and not _direct_artifact_object(request):
        _set(IntentType.INFORMATION_QUERY, "retrieve_knowledge", 0.8,
             "force=request", "content delivery act")
        return intent

    # ── 2e. A request that names nothing to act on ──
    # The utterance is a REQUEST (the act's force) but predicates
    # nothing: no artifact to build, no content to deliver, no action
    # class — the light verb alone ("do this for me") carries the
    # request without saying WHAT.  Guessing an artifact type here
    # would invent the object of the request; the honest reading asks
    # the addressee to supply it.
    if act.force == FORCE_REQUEST and not (act.negated or act.past)             and not _artifact_goal             and not _content_delivery_request(request)             and not investigation             and not act.explanation_request             and not act.predicate_class:
        _set(IntentType.CLARIFICATION, "missing_object_of_request", 0.5,
             "force=request", "the request predicates no action or object")
        intent.scope["missing"] = "object of the request"
        return intent

    # ── 3. Request → artifact production ──
    # A bare instruction fragment ("Given numbers, discard duplicates,
    # ...") is an imperative under ellipsis: an ACTION predicate that is
    # neither a query, a report, a comparison nor a knowledge statement.
    # Equally, when the semantic representation already carries an
    # artifact goal (the parser verified the structure), the content's
    # KIND is an artifact to be produced.
    creating = (act.force == FORCE_REQUEST and not (act.negated or act.past)) \
        or (act.force == FORCE_ASSERT and allows_artifact_creation(act)[0]) \
        or (_artifact_goal and not (act.negated or act.past)
            and not act.social_act)
    if creating:
        intent.topics = _artifact_topics(semantic, request)
        if existing_artifact or goal_value == GoalValue.modify:
            _set(IntentType.CODE_MODIFICATION, "extend_existing_artifact",
                 0.8, "force=request", "existing-artifact evidence")
            return intent
        kind = artifact.get("kind", "") if isinstance(artifact, dict) else ""
        if kind == "project" or goal_value == GoalValue.project:
            _set(IntentType.PROJECT_GENERATION, "create_project", 0.8,
                 "force=request", "artifact=project")
        elif goal_value in (GoalValue.modify,):
            _set(IntentType.CODE_MODIFICATION, "modify_artifact", 0.75,
                 "force=request", "goal=modify")
        else:
            _set(IntentType.CODE_GENERATION, "create_function_or_code",
                 0.8, "force=request", f"artifact={kind or 'function'}")
        intent.scope["artifact_kind"] = kind or "function"
        return intent

    # ── 3. Query → information / capability / advice / analysis ──
    if act.force == FORCE_QUERY:
        # The investigation's PREDICATE decides what the question wants:
        # a query whose predication is a FAILURE asks for a REPAIR
        # ("Why does this error occur?") — the answer is a diagnosis,
        # never a fact about the topic.
        if act.predicate_class == "failure":
            _set(IntentType.REPAIR_REQUEST, "diagnose_failure", 0.8,
                 "force=query", "predicate=failure")
            return intent
        if act.predicate_class == "state":
            # A phatic question about the addressee's state
            _set(IntentType.GREETING, "exchange_greeting", 0.85,
                 "force=query", "predicate=state (phatic)")
            return intent
        if act.predicate_class == "ability" or self_topic:
            if natural_language_topic:
                _set(IntentType.LANGUAGE_CAPABILITY_QUERY,
                     "check_language_support", 0.85,
                     "force=query", "predicate=ability",
                     "topic=natural language")
                return intent
            # A content-delivery request under the polite frame asks for
            # the DISCOURSE's content — an instance or an exposition of
            # what the conversation is about ("Can you give me an
            # example?", its referent inherited from the topic state) —
            # never for the system's abilities.  It falls through to the
            # content branch below; a question about the system itself
            # ("What can you do?") still stands as a capability query.
            # With NO referent anywhere the ask has nothing to apply to,
            # so the honest act asks for the missing target — the
            # capability list answers a different question.
            contentful = (not self_topic
                          and _content_delivery_request(request))
            if contentful:
                if reference.get("reference_source") == "discourse_topic" \
                        or topics:
                    # the discourse (or the utterance) supplies the
                    # referent: the content branch below answers it
                    pass
                else:
                    _set(IntentType.CLARIFICATION, "resolve_missing_target",
                         0.4, "force=query", "content delivery act",
                         "no referent in the utterance or the discourse")
                    intent.scope["missing"] = "target of the request"
                    return intent
            else:
                _set(IntentType.CAPABILITY_QUERY, "describe_capabilities",
                     0.85, "force=query", "predicate=ability",
                     "topic=self")
                return intent
        if act.predicate_class in ("advice", "evaluation"):
            _set(IntentType.RECOMMENDATION_REQUEST,
                 "recommend_or_evaluate", 0.8,
                 "force=query", f"predicate={act.predicate_class}")
            return intent
        if shell_context and act.method_question:
            _set(IntentType.COMMAND_QUERY, "how_to_in_shell", 0.8,
                 "force=query", "method question", "shell context")
            intent.scope["platform"] = knowledge.get("platform", "")
            return intent
        if shell_context and not act.method_question:
            _set(IntentType.COMMAND_QUERY, "shell_topic_query", 0.7,
                 "force=query", "shell context")
            intent.scope["platform"] = knowledge.get("platform", "")
            return intent
        if program_topic:
            _set(IntentType.ANALYSIS, "analyze_program_behavior", 0.7,
                 "force=query", "topic=program behavior")
            return intent
        _set(IntentType.INFORMATION_QUERY, "retrieve_knowledge", 0.75,
             "force=query", "predicate=information request")
        return intent

    # ── 4. Assertion ──
    if act.force == FORCE_ASSERT:
        if act.desiderative and act.predicate_class in ("advice",
                                                        "evaluation"):
            _set(IntentType.RECOMMENDATION_REQUEST,
                 "evaluate_learning_direction", 0.7,
                 "force=assert", "first-person desire",
                 f"predicate={act.predicate_class}")
            return intent
        # An explanation request precedes a comparison predicate: "I
        # don't understand why X and Y differ" asks for a MEANING
        # (the why/comprehension-gap framing), not for a contrast.
        if act.explanation_request:
            style = requested_style(request)
            if style == "simple":
                _set(IntentType.SIMPLIFY_EXPLANATION,
                     "simplify_explanation", 0.7,
                     "force=assert", "predicate=explanation request",
                     "register=simple")
            else:
                _set(IntentType.EXPLANATION_REQUEST, "explain_target", 0.7,
                     "force=assert", "predicate=explanation request")
            return intent
        if act.predicate_class == "comparison":
            _set(IntentType.INFORMATION_QUERY, "compare_topics", 0.75,
                 "force=assert", "predicate=comparison")
            return intent
        if act.method_question and shell_context:
            # A nominal "how to" ask whose topic is a shell environment
            _set(IntentType.COMMAND_QUERY, "how_to_in_shell", 0.7,
                 "force=assert", "method question", "shell context")
            intent.scope["platform"] = knowledge.get("platform", "")
            return intent
        if shell_context and act.method_question:
            _set(IntentType.COMMAND_QUERY, "how_to_in_shell", 0.7,
                 "method question", "shell context")
            return intent
        _set(IntentType.CONVERSATION, "share_or_report", 0.7,
             "force=assert", f"subject={act.subject or 'unspecified'}")
        return intent

    # ── 5. Unclear ──
    _set(IntentType.CLARIFICATION, "resolve_meaning", 0.4,
         "force unknown", "insufficient structure")
    return intent


def _artifact_topics(semantic, request: str) -> list[str]:
    """The content of a creation request comes from the semantic
    representation's entities and objectives — the meaning structure,
    not the raw clause."""
    if semantic is None:
        return []
    names: list[str] = []
    for e in (getattr(semantic, "entities", None) or []):
        n = getattr(e, "name", "")
        if n and n not in names:
            names.append(n)
    for o in (getattr(semantic, "objectives", None) or []):
        n = getattr(o, "direction", "") or getattr(o, "name", "")
        if n and n not in names:
            names.append(n)
    if names:
        return names[:4]
    return extract_topics(request, semantic, None)[:1]


class GoalValue:
    """GoalType value strings (avoids importing the representation
    module in a cycle)."""
    project = "create_project"
    modify = "modify"
    function = "create_function"
    klass = "create_class"
    question = "question"
    chat = "chat"
    explain = "explain"
    analyze = "analyze"
    command_query = "command_query"
    information_query = "information_query"
    ambiguous = "ambiguous"
    unknown = "unknown"
