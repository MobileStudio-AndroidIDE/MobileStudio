"""ConversationKnowledgeEngine: question → intent → knowledge retrieval →
reasoning → ResponsePlan → answer.

UNDERSTAND → RETRIEVE RELEVANT EVIDENCE → REASON → COMPOSE → OUTPUT.

Responsibility separation (the core design):

  Knowledge     = what is known        (the concept store's evidence —
                                        descriptions are EVIDENCE, never
                                        canned answers)
  Reasoning    = what is relevant     (the question's semantic structure
                                        decides the intent and selects the
                                        relevant evidence)
  Response     = what to say now      (a fresh composition from the
                                        evidence, in the input language's
                                        register)

The concept store's aliases/tags are RETRIEVAL HINTS only: they locate
candidate concepts; the semantic structure (the goal type, the
interrogative form — WH-definitional vs polar-functional, the target
reference, the first-person self-statement structure) decides the
dialogue act.  A concept's description is never echoed as the answer:
the evidence is rephrased and composed per the question's intent — a
definition, a simple explanation, a comparison, a suitability
assessment, or a social acknowledgment.

No canned answers: every response is composed from the retrieved
evidence.  False capability claims are never made.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from brain.dialogue_act import requested_style
from brain.knowledge_requirement import analyze_knowledge_requirement
from brain.response_variation import choice, reseed


# ============================================================
# Response plan (the explicit intermediate representation)
# ============================================================

# The words that can NAME something: grammar words are dropped before
# a relevance comparison, so "of", "the" never make two texts agree.
_RELEVANCE_STOP = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "and", "or",
    "but", "with", "from", "into", "over", "under", "about", "for",
    "how", "what", "why", "when", "where", "who", "which", "you",
    "your", "are", "was", "were", "been", "have", "has", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "than",
    "then", "them", "they", "their", "there", "here", "some", "any",
    "all", "one", "two", "more", "most", "other", "very", "also",
    "just", "only", "even", "get", "got", "make", "made", "take",
    "want", "like", "need", "use", "used", "does", "doing", "done",
})


def _content_terms(text: str) -> set[str]:
    """The NAMING words of a text: lower-case, plural-folded, grammar
    dropped — what two texts must share to be about the same thing."""
    words = re.findall(r"[a-zA-Z][a-zA-Z+#']{3,}|[가-힣]{3,}",
                       str(text or ""))
    out = set()
    for word in words:
        low = word.lower()
        if low in _RELEVANCE_STOP:
            continue
        out.add(low.rstrip("s"))
    return out


def _relevance_stems(text: str) -> set[str]:
    """The stems two texts must share to be about the same thing: a
    stem is the word's first five letters, so "program" and
    "programming" name the same subject while unrelated subjects never
    collide."""
    return {term[:5] for term in _content_terms(text) if len(term) >= 5}


# The past's own words: a result titled with them describes how
# something WAS, which answers "history of …" and nothing else.
_PAST_STEMS = {"histo", "versi", "timel", "origi", "past", "evolu",
               "역사", "연혁", "버전", "유래"}
_PAST_TITLE = re.compile(
    r"(?:\b(?:history of|version history|timeline|chronology)\b"
    r"|list of .{0,40}versions?"
    r"|연혁|버전\s*역사|연대기)", re.IGNORECASE)


def _queries_of(requirement) -> list[str]:
    """The parts the requirement recorded — object or trace dict."""
    if requirement is None:
        return []
    if isinstance(requirement, dict):
        raw = requirement.get("queries") or []
    else:
        raw = getattr(requirement, "queries", None) or []
    return [str(q) for q in raw if str(q).strip()]


@dataclass
class ResponsePlan:
    """The response's explicit intermediate representation: the intent,
    the target, the reasoned claims, the answer content, the requested
    language, the style, and the evidence — the same intermediate
    representation the code generation takes through its CodePlan."""
    intent: str = ""             # the dialogue act category
    target: str = ""             # the topic / the question's referent
    claims: list[str] = field(default_factory=list)     # the reasoned claims
    explanation: str = ""        # the knowledge-based explanation content
    answer: str = ""             # the direct answer (the command/capability)
    requested_language: str = "en"  # the realization's register
    style: str = ""              # simple / detailed
    evidence: list[str] = field(default_factory=list)   # the retrieved evidence
    confidence: float = 0.0
    system_name: str = "RENA-BR"   # the identity from the self knowledge
    speech_style: str = "polite"   # the learned register (user memory)
    # The recommendation's option set, STRUCTURED (name, category,
    # description) — the realization composes sentences from this,
    # never a knowledge-dump listing.
    options: list[dict] = field(default_factory=list)
    # The evidence frame: "definition" when the topic IS the evidence's
    # concept ("What is Python?"); "clause" when the topic is the
    # question's own proposition and the evidence only SUPPORTS it —
    # the claim then follows the proposition, never rewrites it as a
    # definition ("Programming languages need different types. <evidence>").
    frame: str = ""
    # The NEIGHBOURING evidence: the related concept's own sentences,
    # kept apart from claims so the realization labels them with the
    # realizing language's own label (never baked in at build time).
    related: list[str] = field(default_factory=list)
    # The turn's CONTEXT: the resolved reference and the topic state
    # the answer was built against (what "that" pointed at, and what
    # the discourse is about).
    reference: dict = field(default_factory=dict)
    topic_state: dict = field(default_factory=dict)
    # The knowledge requirement decided BEFORE retrieval, and the
    # retrieval it produced (evidence with provenance — never the
    # answer itself).
    knowledge_requirement: dict = field(default_factory=dict)
    retrieval: dict = field(default_factory=dict)
    # The turn asked for an INSTANCE of its topic (an example), not its
    # definition: the evidence is sought as the instance the speaker
    # asked for — a definition is never presented as an example.
    instance_request: bool = False
    # The response representation and its sentence plan (what the
    # response says, before it is worded).
    representation: dict = field(default_factory=dict)
    sentence_plan: list = field(default_factory=list)
    # The procedure's STEPS: the evidence's own imperative sentences,
    # in their order — extracted only when there are enough of them to
    # be a procedure.  Descriptive prose never becomes steps, so the
    # response cannot present invented ordering as sourced guidance.
    steps: list = field(default_factory=list)
    # The turn's CONTRAST structure: the pair being compared and the
    # dimensions both sides' evidence speaks to — the axes come from
    # the knowledge graph's own structure (shared facets, shared
    # relations), never invented.  {} when the turn contrasts nothing.
    comparison: dict = field(default_factory=dict)
    # The proposition this turn PUTS FORWARD (a stance act): the claim
    # the speaker holds and the reply discusses from evidence — never
    # the topic the claim merely mentions.  "" for a turn that states
    # no claim of its own.
    proposition: str = ""
    # The response's length budget, decided from the request's own
    # grammar (brain.longform): the bounds the generation obeys.
    budget: dict = field(default_factory=dict)
    # The full material pool for a long-form response: every evidence
    # sentence the turn holds, uncapped — collected only when the
    # budget warrants it.
    longform_material: list = field(default_factory=list)
    # What the long-form generation produced: the sections, the
    # state and the verification report (traced like everything else).
    longform: dict = field(default_factory=dict)
    # The investigation this turn performed: the subject, the workflow
    # the request spelled out (search → evaluate → reformulate →
    # search → synthesize) and how the evidence went — {} for a turn
    # that investigated nothing.
    research: dict = field(default_factory=dict)
    # The request read as a DOCUMENT: one structured request carrying
    # its topic, its knowledge goals (with their dependencies) and its
    # response requirements.  Retrieval and the response's outline are
    # derived from it; a response requirement never becomes a query.
    # None for a turn that asks for no document.
    document: object = None
    # The retrieved, VALIDATED evidence gathered per knowledge goal.
    document_evidence: object = None

    @property
    def web_used(self) -> bool:
        return bool(self.retrieval.get("web_used"))

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "target": self.target,
            "claims": self.claims,
            "explanation": self.explanation,
            "answer": self.answer,
            "requested_language": self.requested_language,
            "style": self.style,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "system_name": self.system_name,
            "speech_style": self.speech_style,
            "options": self.options,
            "frame": self.frame,
            "related": self.related,
            "reference": dict(self.reference),
            "topic_state": dict(self.topic_state),
            "knowledge_requirement": dict(self.knowledge_requirement),
            "retrieval": dict(self.retrieval),
            "instance_request": bool(self.instance_request),
            "representation": dict(self.representation),
            "sentence_plan": list(self.sentence_plan),
            "steps": list(self.steps),
            "comparison": dict(self.comparison),
            "proposition": self.proposition,
            "web_used": self.web_used,
            "budget": dict(self.budget),
            "longform": dict(self.longform),
            "research": dict(self.research),
        }


# ============================================================
# The engine
# ============================================================

# The knowledge concept categories: their activation indicates a
# knowledge question's TOPIC (never a computation request)
_KNOWLEDGE_CATEGORIES = {"programming_language", "language_fundamental"}

# The minimum retrieval score for a concept to count as evidence
_EVIDENCE_MIN_SCORE = 4.0

# The closed grammatical classes and the request's own framing words:
# grammar, never content.  A content word is what a phrase is ABOUT —
# the compatibility test below compares content words only.
_NON_CONTENT_WORDS = frozenset({
    # articles, pronouns, determiners
    "a", "an", "the", "this", "that", "these", "those", "it", "its",
    "they", "them", "their", "he", "she", "we", "you", "i", "me", "my",
    "your", "us", "our", "one", "some", "any", "all", "each", "every",
    # auxiliaries and copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "do",
    "does", "did", "has", "have", "had", "can", "could", "will",
    "would", "shall", "should", "may", "might", "must", "not", "no",
    # conjunctions, prepositions, wh-words
    "and", "or", "but", "if", "then", "than", "so", "because", "as",
    "of", "to", "in", "on", "at", "for", "from", "with", "by", "into",
    "over", "under", "about", "up", "down", "out", "off", "per", "via",
    "what", "who", "which", "when", "where", "why", "how", "there",
    # the request's own framing and predicate modifiers: HOW the ask is
    # phrased, never WHAT it is about
    "please", "tell", "give", "show", "let", "lets", "make", "create",
    "write", "explain", "describe", "define", "help", "find", "check",
    "compare", "list", "print", "return", "keep", "remove", "sort",
    "filter", "use", "using", "get", "set", "need", "needs", "want",
    "more", "less", "very", "really", "just", "also", "simply",
    "like", "well", "ok", "okay", "yes", "hi", "hello",
    "hey", "now", "then", "actually", "something", "anything",
    "everything", "next", "last", "first", "again",
})


def _content_words(text: str) -> list[str]:
    """The content words of a phrase: lowercase, the simple English
    plural folded ("algorithms" → "algorithm"), the Korean case
    particles shed, closed classes and request framing dropped.
    Comparing content words is how the topic and the evidence are
    matched against each other."""
    words: list[str] = []
    for raw in re.findall(r"[a-z][a-z+#]{1,}|[가-힣]{2,}", (text or "").lower()):
        word = raw
        if word in _NON_CONTENT_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if "가" <= word[:1] <= "힣":
            stripped = re.sub(
                r"(은|는|이|가|을|를|에|의|도|로|으로|에서|과|와|만|"
                r"부터|까지|랑|이랑)$", "", word)
            word = stripped or word
        words.append(word)
    return words


def _sentences(text: str) -> list[str]:
    """Evidence prose split into its sentences — the reasoning's claim
    granularity.  A concept description is a paragraph; the response
    composes from sentences of it, so 'a simpler way' can ask for fewer
    and 'in detail' for more, without ever echoing the paragraph."""
    parts = [p.strip() for p in re.split(
        r"(?<=[.!?])\s+", (text or "").strip())]
    return [p for p in parts if p]


def _claim_budget(style: str) -> int:
    """How many evidence sentences a claim is composed of: the simple
    register states the core, the brief one states the same evidence in
    fewer sentences, the detailed one develops it — none of them dumps
    the source paragraph."""
    return 1 if style in ("simple", "brief") else 2


def _order_procedure(plan: ResponsePlan) -> None:
    """Extract the procedure's STEPS from the evidence: a step is the
    source's own imperative sentence — it TELLS the reader what to do.

    An ordered run of them is the procedure the request asked for.
    Descriptive prose is not a procedure, so it stays in `claims` and
    the response presents it as what it is: the guidance the evidence
    supports, never steps the evidence never gave."""
    steps = [s for s in plan.claims if _is_step(s)]
    if len(steps) >= 2:
        plan.steps = steps
        plan.frame = "procedure" if not plan.frame else plan.frame


def _is_step(sentence: str) -> bool:
    """The sentence STATES AN ACTION TO TAKE — its form is imperative."""
    try:
        from brain.dialogue_act import analyze_act, FORM_IMPERATIVE
        return analyze_act(sentence).form == FORM_IMPERATIVE
    except Exception:
        return False


def _core(sentence: str) -> str:
    """The simple register's claim FORM: the sentence's core — its
    statement up to the first elaboration, when it carries one.  The
    detailed register develops the evidence; the simple one states it."""
    head = re.split(r"\s*[:;]\s+", sentence, maxsplit=1)[0]
    head = head.rstrip(",—- ").strip()
    if 40 <= len(head) < len(sentence.strip()):
        if head[-1] not in ".!?。！？":
            head += "."
        return head
    # No colon/apposition to take the core from: a dependent clause is
    # the elaboration — "X so … that Y", "X because Y" — so the simple
    # register states X, the same statement without the development.
    # The detailed register keeps the dependency; the simple one does
    # not.
    dep = re.split(
        r"\s+(?:because|since|although|though|while|whereas|"
        r"so\s+\w+(?:ly)?\s+that|such\s+\w+\s+that|"
        r"in\s+order\s+to)\s+|"
        r",\s+(?:because|since|although|though|so\s+that)\s+|"
        r",\s+(?:making|giving|causing|leaving|resulting|leading"
        r"|rendering|providing|allowing)\s+|"
        r"(?:때문에|그래서|하지만)",
        sentence, maxsplit=1, flags=re.IGNORECASE)[0]
    dep = dep.rstrip(",—- ").strip()
    if dep and dep != sentence.strip() and len(dep) >= 25:
        if dep[-1] not in ".!?。！？":
            dep += "."
        return dep
    # A sentence that STATES TWO THINGS states its first — the
    # coordinating clause is the elaboration, so "X, and Y" is simply
    # X in the simple register.  Without this the simpler render would
    # be word for word the detailed one.
    clause = re.split(r",\s+(?:and|but|or|so)\s+", sentence,
                      maxsplit=1, flags=re.IGNORECASE)[0]
    clause = clause.rstrip(",—- ").strip()
    if clause and clause != sentence.strip() and len(clause) >= 40:
        if clause[-1] not in ".!?。！？":
            clause += "."
        return clause
    return sentence

# The dialogue-act layer's intent (the ACT the speaker performs) → the
# plan's dialogue-act category.  The act decides the category; the
# content's kind may REFINE it within the same class (a suitability
# assessment is a recommendation about a specific option; an analysis
# request is a query about a program).
_DIALOGUE_TO_PLAN = {
    "greeting": "greeting",
    "conversation": "conversation",
    "information_query": "information_query",
    "explanation_request": "explanation_request",
    "simplify_explanation": "simplify_explanation",
    "capability_query": "capability_query",
    "language_capability_query": "language_capability_query",
    "recommendation_request": "recommendation_request",
    "procedural_instruction": "procedural_instruction",
    "command_query": "command_query",
    "analysis": "analysis_request",
    "clarification": "clarification",
    # The investigation is its own act category: the knowledge is not
    # in the conversation yet, so evidence is sought and iterated.
    "research": "research",
    # The stance act: the turn's own claim is what is discussed, so the
    # category is never re-cast as a definition of the topic it names.
    "opinion_request": "opinion_request",
    # The proposal act: the turn asked for a candidate to be given, so
    # the category is never re-cast as a definition either.
    "proposal_request": "proposal_request",
}

# The content-dependent refinements the engine may keep when they belong
# to the same act class as the dialogue act's category.
_PLAN_REFINE = {
    "greeting": {"greeting", "salutation", "state_inquiry", "gratitude",
                 "valediction"},
    "conversation": {"conversation", "greeting", "salutation",
                     "state_inquiry", "gratitude", "valediction",
                     "identity"},
    "information_query": {"information_query", "comparison",
                          "suitability_query", "analysis_request",
                          "identity", "explanation_request"},
    "explanation_request": {"explanation_request", "information_query",
                            "simplify_explanation"},
    # The simplifying explanation IS the act's own category: the
    # content layer may not widen it back into a plain explanation
    # (the register the speaker asked for survives).
    "simplify_explanation": {"simplify_explanation"},
    # A procedure request IS the act's own category: the content layer
    # may not re-cast it as a definition or a conversation — the
    # ordered way of doing something is what was asked for.
    "procedural_instruction": {"procedural_instruction"},
    "capability_query": {"capability_query", "identity"},
    "language_capability_query": {"language_capability_query",
                                  "capability_query", "identity"},
    # A recommendation is an ACT about alternatives.  The content layer
    # may refine it to a property assessment or an ability check, but it
    # must never be re-cast as a definition: "where should I begin?" is
    # not answered by restating the topic.
    "recommendation_request": {"suitability_query", "capability_query",
                               "identity"},
    "analysis_request": {"analysis_request", "analysis_request"},
    # An investigation IS the act's own category: the content layer
    # may not re-cast it as a definition — what was asked for is the
    # finding-out, not a restatement of what is already stored.
    "research": {"research"},
    "command_query": set(),
    "clarification": set(),
    # The stance act IS the act's own category: the content layer may
    # not re-cast a claim under discussion as a definition — what was
    # asked for is a position on the claim, not an encyclopedia entry.
    "opinion_request": {"opinion_request"},
    # The proposal act IS the act's own category: the content layer may
    # not re-cast a candidate that was asked for as a definition.
    "proposal_request": {"proposal_request"},
}

# The act layer's own GOAL for an interaction whose kind it already
# named ("present_self"): the content layer reads that decision instead
# of deriving a weaker one from the surface form.  The act's goal is
# the authority on the act — the utterance's wording never decides it.
_ACT_GOAL_CATEGORY = {
    "present_self": "identity",
}


class ConversationKnowledgeEngine:
    """Question → intent → knowledge retrieval → reasoning → answer."""

    def __init__(self, concept_store, relation_store, self_knowledge,
                 capability_registry, conversation_history=None):
        self.concepts = concept_store
        self.relations = relation_store
        self.self_knowledge = self_knowledge
        self.web_enabled = True  # the web as an external evidence source
        self.capabilities = capability_registry
        self.conversation_history = conversation_history or []
        # The topic state's provider (the conversation manager): the
        # stable current topical referent an anaphoric reference
        # resolves against — a reset retires it.
        self.topic_provider = None

    # ------------------------------------------------------------------
    # Intent inference (from the semantic structure)
    # ------------------------------------------------------------------

    # The conventional expressive acts and their utterance forms: for a
    # bare expressive (goal chat, no predication) the act expression's
    # meaning IS the classification (the speech-act layer's knowledge —
    # the response is still composed: identity, register, variation).
    _EXPRESSIVE_ACTS = (
        ("gratitude", ("thanks", "thank you", "thx", "cheers",
                       "감사", "고마워", "고맙")),
        ("valediction", ("goodbye", "good bye", "bye", "see you",
                         "farewell", "later", "잘가", "잘 가", "안녕히 가")),
        ("salutation", ("hello", "hi", "hey", "good morning",
                        "good afternoon", "good evening",
                        "안녕하세요", "안녕", "반갑")),
        ("identity", ("자기소개", "introduce yourself",
                      "who are you", "what are you",
                      "너 누구", "당신 누구")),
    )

    # The state inquiry: a phatic question about how the ADDRESSEE is
    # doing — the grammatical subject is the interlocutor ("How are
    # you?", "기분 어때?"), so it is dialogue about self-state, not a
    # knowledge question about an external topic.
    _STATE_INQUIRY = re.compile(
        r"how\s+(?:are|r)\s+(?:you|u)|"
        r"how'?s\s+(?:it\s+going|things)|"
        r"what'?s\s+up\b|"
        r"기분|잘\s*지내|지내(?:어|냐|니|요)\b|어떻게\s*지내|"
        r"뭐하고\s*(지내|있어|사는)|"
        r"밥\s*먹|did\s+you\s+eat|had\s+(?:lunch|breakfast|dinner)",
        re.IGNORECASE)

    def _chat_act(self, utterance: str) -> str:
        """The dialogue act of a bare expressive utterance: gratitude /
        valediction / salutation — classified from the act expression's
        meaning (not a GoalType keyword list — the goal is already
        chat)."""
        u = utterance.strip().lower().rstrip("!?.~")
        for act, forms in self._EXPRESSIVE_ACTS:
            for form in forms:
                if u == form:
                    return act
                if u.startswith(form):
                    rest = u[len(form):]
                    if not rest:
                        return act
                    # English words need a boundary; Korean attaches
                    # its suffixes directly to the act expression
                    if form.isascii():
                        if rest[0] in " ,!?'\"-":
                            return act
                    else:
                        return act
        if self._STATE_INQUIRY.search(u):
            return "state_inquiry"
        return "conversation"

    def infer_intent(self, question: str, semantic) -> str:
        """The plan's dialogue-act category.

        The DIALOGUE ACT (the act layer's intent type) is the authority
        on the ACT; the content's structure may REFINE it within the
        same act class.  Neither the topic words nor a question→answer
        table decide the category.
        """
        act_intent = self._intent_from_act(semantic)
        content_intent = self._intent_from_content(question, semantic)
        if not act_intent:
            return content_intent
        refine = _PLAN_REFINE.get(act_intent)
        if refine is None:
            return act_intent
        if content_intent in refine:
            return content_intent
        return act_intent

    @staticmethod
    def _intent_from_act(semantic) -> str:
        """The dialogue-act layer's intent type for this utterance."""
        di = getattr(semantic, "dialogue_intent", None) or {}
        dtype = str(di.get("type") or "").strip()
        return _DIALOGUE_TO_PLAN.get(dtype, "")

    @staticmethod
    def _interrogative_clause(question: str) -> str:
        """The utterance's interrogative sentence.

        The question acts are read from the sentence that actually
        asks: "I want to build apps. Is that realistic for a beginner?"
        is a polar question whose subject is the leading statement.
        """
        if "?" not in question and "？" not in question:
            return question
        found = re.findall(r"[^.!?\n]*[?？]", question)
        return found[-1].strip() if found else question

    def _intent_from_content(self, question: str, semantic) -> str:
        """The dialogue act kind from the semantic representation's
        structure — never a keyword mapping:

          - the goal EXPLAIN → an explanation request
          - a first-person self-statement ("I am learning Python") → a
            conversation act (a social acknowledgment, no knowledge dump)
          - a self-referential target (the question's topic is the
            system's own abilities) → a capability query
          - a language-topic polar question ("Can you speak English?") →
            a language capability query (an honest support check)
          - a WH-definitional interrogative ("What is X?") with a
            knowledge concept target → an information query
          - a polar interrogative about a program's behavior ("Does this
            program work?") → an analysis request (the honest ask-for-
            the-code routing)
          - the goal CHAT with a greet predication → a greeting
          - otherwise → a conversation act
        """
        goal = getattr(semantic, 'goal', None)
        goal_type = getattr(goal, 'goal_type', None)
        goal_type_value = goal_type.value if goal_type is not None else ""
        target = (getattr(goal, 'target', '') or "").strip()
        target_lower = target.lower()
        question_lower = question.lower()
        # The structural checks below read the INTERROGATIVE CLAUSE, not
        # the whole utterance: "I want to build apps. Is that realistic
        # for a beginner?" asks a polar question in its second sentence,
        # and the leading statement must not swallow it.
        _clause = self._interrogative_clause(question)
        struct_lower = _clause.lower() if _clause is not question \
            else question_lower

        if goal_type_value == "explain":
            return "explanation_request"

        # The ACT LAYER already named this interaction's kind (its own
        # goal: "present_self").  That decision is the authority — the
        # content layer refines it into the plan's category instead of
        # re-deriving a weaker one from the surface wording, so the
        # turn never falls back to a generic acknowledgment.
        act_goal = str((getattr(semantic, "dialogue_intent", None)
                        or {}).get("goal") or "")
        if act_goal in _ACT_GOAL_CATEGORY:
            return _ACT_GOAL_CATEGORY[act_goal]

        # The social acts (salutation/gratitude/valediction) BEFORE any
        # knowledge retrieval: a bare expressive utterance ("Hello",
        # "Thanks!", "Goodbye") is an interaction, not a question about
        # a concept — the literal mention ('Hello' is a tag of the
        # Literal Value data concept) is a vocabulary echo, not a topic.
        # The goal CHAT decides the social KIND; the utterance's
        # semantic content (the act expression itself) decides the act.
        if goal_type_value == "chat":
            return self._chat_act(question_lower)

        # The state inquiry: a phatic question about how the addressee
        # is doing ("How are you?", "오늘 기분 어때?") — dialogue about
        # self-state routed as chat, never external knowledge retrieval.
        if self._STATE_INQUIRY.search(question_lower):
            return "state_inquiry"

        # The first-person self-statement: the sentence's agent is the
        # user speaking about themselves ("I am learning X") — a social
        # sharing act, not an information request (no knowledge dump)
        if re.match(r"^(i\s+am|i'm|i\s+|저는\s|나는\s)", struct_lower):
            return "conversation"

        # The self-referential target: the question's topic is the
        # system itself (resolved against the self knowledge) — a
        # language support check, an ability question, or the identity
        # itself ("Who are you?" asks BEING, not ability)
        if self._is_self_referential(target_lower, struct_lower):
            # A polar question about a LANGUAGE the system speaks → the
            # language capability check
            if self._mentions_natural_language(
                    target_lower + " " + struct_lower):
                return "language_capability_query"
            # The grammatical predicate decides: an ABILITY predicate
            # (can / able / 할 수) → what the system DOES; otherwise the
            # question asks what the system IS → the identity act
            if self._has_ability_predicate(struct_lower):
                return "capability_query"
            return "identity"

        # A polar interrogative: the yes/no question forms decide by
        # their TOPIC (never keywords about the answer)
        if re.match(r"^(does|do|is|are|will|would|can|could)\b",
                    struct_lower):
            # A polar question about the system's own ability (the
            # self-referential semantics: the subject is "you" — the
            # system) with a natural-language mention → the language
            # capability check (the honest support check)
            if re.search(r"\byou\b|너|시스템", struct_lower) and \
                    self._mentions_natural_language(struct_lower):
                return "language_capability_query"
            if re.search(r"\byou\b|너", struct_lower) and \
                    self._is_self_referential(target_lower, struct_lower):
                return "capability_query"
            # A polar question about a program's behavior → an
            # analysis request ("Does this Python program work?")
            if re.search(r"(program|code|script|이\s*프로그램|이\s*코드)",
                         target_lower + " " + struct_lower):
                return "analysis_request"
            # A polar question about a concept's property ("Is Python
            # suitable for web development?") → an evidence-based
            # suitability assessment
            if self._find_evidence_concepts(self._clean_topic(target)):
                return "suitability_query"

        # A WH-definitional interrogative with a knowledge concept
        # target → an information query
        if self._find_evidence_concepts(target):
            return "information_query"

        # The WH-definitional grammar decides even when the LOCAL
        # knowledge is silent — the web is the external evidence
        # source behind the concept store
        if re.match(r"^(what|who|whom|whose|which)\s+"
                    r"(is|are|was|were|do|does|did)\b",
                    struct_lower) or re.search(
                r"(뭐|무엇)(야|예요|일까|인가|이지)", struct_lower):
            return "information_query"

        if goal_type_value == "chat":
            return "greeting"
        return "conversation"

    def is_self_question(self, request: str, semantic) -> bool:
        """The question's referent is the SYSTEM itself (the addressee
        reference: you / 너 / yourself / the system's own name) — such a
        question is dialogue about the self, routed as chat, never as
        knowledge retrieval about an external topic."""
        goal = getattr(semantic, 'goal', None)
        target = ((getattr(goal, 'target', '') or "") + " " + request).lower()
        # The phatic state inquiry addresses the interlocutor by the
        # second-person subject understood ("How are you?") — the
        # addressee is implicit, so no explicit you/너 is required.
        if self._STATE_INQUIRY.search(request):
            return True
        has_addressee = bool(re.search(
            r"\byou\b|\byour\b|yourself|\bself\b|너\b|당신|자기",
            target))
        identity = self.self_knowledge.get_identity() if self.self_knowledge else {}
        name = str(identity.get("name", "") or "").lower()
        if not has_addressee and not (name and name in target):
            return False
        return self._is_self_referential(
            ((getattr(goal, 'target', '') or "")).lower(), request.lower())

    def _is_self_referential(self, target_lower: str,
                             question_lower: str) -> bool:
        """The question's topic is the system itself: the self
        knowledge's identity/capabilities resolve the target."""
        capability_terms = ["can you do", "할 수 있", "할수 있", "뭘 할",
                            "무엇을 할", "your abilit", "기능",
                            "what are you", "who are you", "너 누구",
                            "뭐야", "어떤 ai", "어떤 모델", "어떤 챗봇",
                            "어떤 agent", "어떤 채봇", "어떤 시스템",
                            "what kind of", "what sort of", "what type of",
                            "tell me about you", "your name", "너는 어떤"]
        if any(term in target_lower or term in question_lower
               for term in capability_terms):
            return True
        identity = self.self_knowledge.get_identity() if self.self_knowledge else {}
        ai_name = str(identity.get("name", "")).lower()
        if ai_name and ai_name in target_lower:
            return True
        return False

    @staticmethod
    def _has_ability_predicate(text_lower: str) -> bool:
        """The utterance predicates ABILITY (what the system can DO)
        rather than BEING (what the system IS) — the grammatical
        predicate decides the act, not a keyword answer table."""
        return bool(re.search(
            r"\b(can|could|able to|manage to|do you (?:do|handle|"
            r"support|provide))\b"
            r"|할\s*수\s*(있|있어|있는)|할수\s*(있)|뭘\s*할|"
            r"무엇을\s*할|할\s*줄|기능",
            text_lower))

    @staticmethod
    def _mentions_natural_language(text_lower: str) -> bool:
        """A natural-language mention (the human languages, not the
        programming languages) — the language capability topic."""
        natural = ["english", "korean", "japanese", "chinese",
                   "spanish", "french", "german",
                   "영어", "한국어", "일본어", "중국어"]
        return any(lang in text_lower for lang in natural)

    # ------------------------------------------------------------------
    # Knowledge retrieval (the relevant evidence only)
    # ------------------------------------------------------------------

    def _find_evidence_concepts(self, target: str) -> list:
        """The retrieval: the concept store's scored search — the
        aliases/tags are retrieval hints.  Only the strong matches
        (≥4.0: an exact tag or a multi-word tag hit) count as
        evidence; weak noise never becomes knowledge.

        The domain validation: the question's context decides the
        evidence's domain — a question with NO shell context (no shell
        platform/language/operation concepts in the question) ranks the
        shell-domain concepts BELOW the general knowledge concepts
        ("What is a variable?" is a programming-fundamentals question,
        not a shell-scripting one; the shell_variable concept is the
        evidence only when the question's context is shell-related).
        """
        if not target:
            return []
        scored = self.concepts.search_with_scores(target)
        # The context validation: the question's own concepts decide
        # whether the shell domain is the evidence's domain
        has_shell_context = self._has_shell_context()
        evidence = []
        for c, s in scored:
            if s < _EVIDENCE_MIN_SCORE:
                continue
            if c.category.startswith("shell") and not has_shell_context:
                # A shell concept is not the evidence for a question
                # whose context is not shell-related — demote below the
                # general knowledge concepts
                continue
            evidence.append((c, s))
        # The fallback: when the shell demotion removed everything, the
        # general knowledge concepts alone are the evidence
        if not evidence:
            evidence = [(c, s) for c, s in scored
                        if s >= _EVIDENCE_MIN_SCORE
                        and not c.category.startswith("shell_")]
        return evidence

    def _has_shell_context(self) -> bool:
        """The question's context is shell-related: the question's own
        semantic content activates the shell platform/language/operation
        concepts (the conversation history's shell topics count too)."""
        # The engine's per-question context: the question's activated
        # concepts are passed by the handler (the engine reads them
        # from the question-level activation done at build time)
        return bool(getattr(self, '_question_shell_context', False))

    def set_question_context(self, shell_context: bool) -> None:
        """Set the question's shell context (per-question, set by the
        handler before the retrieval)."""
        self._question_shell_context = bool(shell_context)

    @staticmethod
    def _compatible_evidence(concept, target: str) -> bool:
        """Whether this evidence is ABOUT the target — the semantic
        compatibility check the retrieval runs before a concept becomes
        an answer's basis.

        Two structural readings decide it, never a keyword test:
          the topic NAMES the concept   — every content word of the
                                          concept's name appears in the
                                          topic ("What is Python?" →
                                          Python),
          the topic's HEAD is the concept — the topic's last content
                                          word is one of the concept's
                                          own words (its name or its
                                          tags) ("the average" →
                                          Calculate Average).
        A concept that satisfies neither scored a hit without being
        about the topic: "programming languages" is not "Software
        Project", whatever the scorer says."""
        topic_words = _content_words(target)
        if not topic_words:
            return True
        name_words = _content_words(concept.name)
        if name_words and all(w in topic_words for w in name_words):
            return True
        identity = set(name_words)
        for tag in (concept.tags or []):
            identity.update(_content_words(tag))
        if not identity:
            return False
        return topic_words[-1] in identity

    def _retrieve_topic(self, topic: str) -> tuple:
        """Retrieve a topic's concept evidence + the related concepts.

        Every candidate passes the semantic compatibility check first:
        evidence is what the retrieval found ABOUT the topic, never
        merely what it matched."""
        scored = [pair for pair in self._find_evidence_concepts(topic)
                  if self._compatible_evidence(pair[0], topic)]
        if not scored:
            return None, []
        top = scored[0][0]
        related = self.relations.find_related(top.id, direction="both") \
            if self.relations else []
        related_concepts = []
        for rel in related[:3]:
            other = self.concepts.get(rel.target_id
                                      if rel.source_id == top.id
                                      else rel.source_id)
            if other is not None:
                related_concepts.append(other)
        return top, related_concepts

    # ------------------------------------------------------------------
    # Knowledge requirement (decided BEFORE any retrieval)
    # ------------------------------------------------------------------

    def _local_knowledge_available(self, intent: str, target: str) -> bool:
        """Whether the LOCAL knowledge the act answers from exists.

        The recommendation answers from the choice categories (its own
        knowledge source); an explanation/definition answers from the
        concept store; the identity/capability acts answer from the self
        knowledge — each checked against the source that act reads."""
        if intent in ("identity", "capability_query",
                      "language_capability_query"):
            return bool(self.self_knowledge)
        if intent in ("recommendation_request", "suitability_query"):
            if self._find_evidence_concepts(target):
                return True
            return bool(getattr(self, "_OPTION_CATEGORIES", ()))
        if not self._topic_contentful(target):
            return False
        return bool(self._find_evidence_concepts(target))

    def _analyze_requirement(self, plan: "ResponsePlan", semantic=None,
                             question: str = "", target: str = "",
                             local_available: bool | None = None,
                             intent: str = ""):
        """Run the knowledge requirement analysis for this turn and
        record it on the plan (the trace reads the SAME decision the
        retrieval acts on)."""
        effective = str(intent or plan.intent or "")
        if local_available is None:
            local_available = self._local_knowledge_available(
                effective, target)
        reference = getattr(self, "_turn_reference", None)
        if not isinstance(reference, dict):
            reference = {}
        if semantic is None:
            semantic = getattr(self, "_turn_semantic", None)
        req = analyze_knowledge_requirement(
            semantic, question, reference,
            getattr(self, "_turn_topic_state", None),
            local_available=local_available, target=target,
            intent=effective,
            instance=str(getattr(self, "_turn_instance_noun", "") or ""))
        plan.knowledge_requirement = req.to_dict()
        return req

    # ------------------------------------------------------------------
    # Reasoning → ResponsePlan
    # ------------------------------------------------------------------

    def build_plan(self, question: str, semantic,
                   language: str = "en",
                   user_profile: dict | None = None) -> ResponsePlan:
        """Build the ResponsePlan: the intent → the knowledge retrieval
        (the relevant evidence only) → the reasoning (the evidence →
        the claims composed per the intent) → the answer content.

        user_profile carries the learned preferences (the self-improvement
        loop's output — speech_style, explanation_length): the profile
        sets the register the response is realized in, while an explicit
        adverbial in THIS request ("in simple terms") wins over the
        standing preference."""
        intent = self.infer_intent(question, semantic)
        goal = getattr(semantic, 'goal', None)
        target = (getattr(goal, 'target', '') or "").strip()
        # The turn's CONTEXT: the resolved reference (what the
        # utterance's referring expression points AT) and the topic
        # state it resolves against.  A reference that did not resolve
        # asks for clarification — it is never sent to a retriever.
        reference = dict(getattr(semantic, "reference", None) or {})
        self._turn_reference = reference
        self._turn_semantic = semantic
        try:
            self._turn_topic_state = self.topic_provider.topic_state                 if self.topic_provider is not None else None
        except Exception:
            self._turn_topic_state = None
        # An unresolved reference: the referent is what is missing, so
        # the act asks for it.  The clarification is decided by the
        # RESOLUTION itself — whether or not the expression has a
        # surface form (a content act with no referent anywhere asks
        # back just as an anaphor with no antecedent does).
        if reference.get("needs_clarification"):
            intent = "clarification"
        # The style: an adverbial that asks for LESS of the evidence
        # ("in a simpler way", "briefly") → simple.  This is the
        # request's own framing, the register cue the speaker chose.
        q_specified_simple = requested_style(question) == "simple"
        style = "simple" if q_specified_simple else "detailed"
        profile = user_profile or {}
        speech_style = str(profile.get("speech_style") or "polite").strip()
        explanation_length = str(
            profile.get("explanation_length") or "").strip().lower()
        # The learned explanation-length preference sets the standing
        # style — unless this request explicitly asked for simple.
        # "brief" is the preference (fewer sentences of the same
        # evidence); "simple" is THIS request's own ask, which also
        # trims the claim to its core.  The speaker's present request
        # and their standing preference are different signals.
        if not q_specified_simple:
            if explanation_length in ("short", "shorter", "concise",
                                      "brief", "간결", "짧게"):
                style = "brief"
            elif explanation_length in ("long", "longer", "detailed",
                                        "자세하게"):
                style = "detailed"
        plan = ResponsePlan(intent=intent, target=target,
                           requested_language=language, style=style,
                           speech_style=speech_style or "polite")
        # The dialogue-act layer's content topics: the referent of a
        # query/report is its topic, never the sentence.  A RESOLVED
        # reference wins: it is where the discourse decided the
        # expression points.
        topics = list(getattr(semantic, "intent_topics", None) or [])
        # the reference is resolved when it names a referent and did
        # not ask back — the same reading the requirement analysis uses
        if reference.get("target") and not reference.get(
                "needs_clarification"):
            topics = [str(reference["target"])] + [
                t for t in topics
                if str(t).strip().lower()
                != str(reference["target"]).strip().lower()]
        plan.reference = dict(reference)
        # A PAIR reference carries its counterpart: the turn asks about
        # BOTH members ("How is that different from the other one?"), so
        # the plan's target names the pair and the reasoning composes a
        # contrast from both members' evidence.
        _contrast = str(reference.get("contrast_with") or "").strip()
        if _contrast and reference.get("target") and \
                _contrast.strip().lower() != \
                str(reference.get("target")).strip().lower():
            topics = [str(reference["target"]), _contrast]
            # The pair reference IS the comparison's structure: both
            # members, named by the discourse itself.
            plan.comparison = {
                "subject_a": str(reference.get("target")).strip(),
                "subject_b": _contrast,
                "dimensions": [],
            }
        if topics:
            # A topic that is a fragment of a proper name is restored
            # to the full name the request states (the pronoun "it" in
            # "What is Circuit of the Americas and why is it …" once
            # resolved to the substring fragment "Circu").
            try:
                from brain.dialogue_act import (
                    _restore_proper_phrases as _restore_names)
                topics = _restore_names(list(topics), question)
            except Exception:
                pass
            plan.target = " and ".join(topics) if len(topics) >= 2 \
                else topics[0]
        # The topic is CONTENT under the question's framing: the
        # grammatical interrogative is stripped ("Python이 뭐야" →
        # "Python"), exactly as the previous-topic resolution does.
        if plan.target:
            plan.target = self._topic_from_question(plan.target)
        # A goal that carried the whole sentence would be sent to the
        # retriever AS the sentence: the query is the topic the act
        # layer extracted, never the request's own text (#10).
        if len(plan.target) > 60:
            import re as _re
            from brain.dialogue_act import extract_topics as _topics_of
            _found = [t for t in (_topics_of(question) or []) if t]
            if not _found:
                for _sentence in _re.split(r"(?<=[.!?])\s+", question):
                    _found = [t for t in (_topics_of(_sentence) or []) if t]
                    if _found:
                        break
            if _found:
                # the noun phrase ends where the request's own list or
                # elaboration begins: the query names the topic
                _head = _re.split(
                    r",\s*(?:including|with|covering|and how|such as)\b|"
                    r"\s+(?:including|covering|such as)\s+",
                    _found[0], maxsplit=1, flags=_re.IGNORECASE)[0]
                plan.target = _head.strip()[:100] or _found[0][:100]
        # A knowledge query with no content topic ("왜?", a fresh
        # "그거 어때?") is a clarification act — retrieving with the
        # sentence itself as query would answer with whatever the
        # knowledge layer happens to match: a dump about the wrong
        # thing (#10).
        if intent in ("information_query", "explanation_request",
                      "simplify_explanation", "procedural_instruction") \
                and not self._topic_contentful(plan.target):
            intent = "clarification"
            plan.intent = "clarification"
            if plan.target.strip().lower().rstrip("??.!") \
                    in self._BARE_INTERROGATIVES:
                # no referent to confirm — a bare interrogative asks
                # for its missing content, not for confirmation of
                # the word itself.
                plan.target = ""
        # The identity from the self knowledge (composed into the
        # greeting, never hardcoded in the realization)
        try:
            identity = self.self_knowledge.get_identity() \
                if self.self_knowledge else {}
            name = str(identity.get("name", "") or "").strip()
            if name:
                plan.system_name = name
        except Exception:
            pass

        # The INSTANCE ask: the speaker wants an OCCURRENCE of the
        # topic (an example), not its definition.  The ask's own noun
        # and the act's form (an ask, never an assertion) decide —
        # recorded on the plan so the evidence layer seeks the instance
        # the request asked for instead of presenting the definition
        # as if it were one.
        instance_noun = self._instance_request(question, semantic)
        self._turn_instance_noun = instance_noun
        plan.instance_request = bool(instance_noun)

        # DOCUMENT-LEVEL SEMANTIC ANALYSIS — the whole request read as
        # ONE structured request before any requirement is derived from
        # it: its topic, its knowledge goals (each carrying what it
        # depends on) and its response requirements.  Local references
        # are resolved here, so a pronoun never reaches a retriever and
        # a response requirement ("clear sections") never becomes a
        # query.  The requirement analysis and every retrieval below
        # read the structured request, not the surface text.
        structured = None
        try:
            from brain.document_semantics import analyze_document
            structured = analyze_document(
                question, semantic,
                list(getattr(semantic, "intent_topics", None) or []))
        except Exception:
            structured = None
        if structured is not None and structured.knowledge_goals:
            plan.document = structured
            # a segment of the discourse target that only a response
            # shaping clause produced describes the REPLY ("Give a
            # detailed"), never the subject, so it never names what
            # the response is about
            plan.target = self._content_target(plan.target, structured)
            # the document's subject only takes the target when the
            # request's own extraction found no content topic: the
            # discourse target stays the discourse's
            if self._topic_contentful(structured.topic) and not \
                    self._topic_contentful(plan.target):
                plan.target = structured.topic
        else:
            structured = None

        # KNOWLEDGE REQUIREMENT ANALYSIS — decided from the act, the
        # resolved reference and the local knowledge probe, BEFORE any
        # retrieval.  It records whether the web is needed and why; the
        # builders below act on it.
        plan.knowledge_requirement = self._analyze_requirement(
            plan, semantic, question, plan.target, intent=intent).to_dict()

        # An unresolved reference: the referent is what is missing, so
        # the ask-back names what is missing — never the request's own
        # text (echoing it would answer nothing).
        if intent == "clarification" and reference.get(
                "needs_clarification"):
            plan.target = ""

        # The OUTPUT BUDGET: the response's length bounds decided from
        # THIS request's grammar BEFORE any retrieval happens.  The
        # evidence layer reads the SAME bounds (a deep turn asks its
        # sources for more before it stops) and the register follows
        # what THIS request asked for — a standing preference never
        # overrides an explicit depth ask, and an explicit depth ask
        # never collapses to the profile's brevity.
        try:
            from brain.longform import (
                build_response_specification, compute_response_budget,
                document_budget)
            budget = compute_response_budget(question, plan, semantic)
            if structured is not None:
                # the document's own depth: what the request asked for
                # in length, not the register's guess at it
                budget = document_budget(structured, budget)
            plan.budget = budget.to_dict()
            if budget.warranted:
                plan.style = "detailed"
            elif budget.reason == "brief register":
                plan.style = "brief"
            # The RESPONSE SPECIFICATION: the same decision, written
            # once where the understanding can carry it — the
            # generation reads it, so nothing re-derives the ask.
            spec = build_response_specification(question, plan, budget)
            if budget.reason.endswith("+document"):
                # the specification and the budget are the SAME decision
                spec["target_length"] = budget.target
                spec["minimum_length"] = budget.minimum
                spec["maximum_length"] = budget.maximum
                spec["warrants_longform"] = True
            spec["detail_level"] = plan.style
            spec["explanation_style"] = plan.style
            # the request's own response-shaping units travel with it
            previous_spec = getattr(semantic, "response_spec", None)
            carried = list(getattr(previous_spec, "constraints", None)
                           or [])
            merged = list(carried)
            for constraint in spec["constraints"]:
                if constraint not in merged:
                    merged.append(constraint)
            spec["constraints"] = merged
            from brain.semantic_representation import ResponseSpecification
            semantic.response_spec = ResponseSpecification(**spec)
        except Exception:
            plan.budget = dict(plan.budget or {})

        if intent == "identity":
            self._build_identity_plan(plan)
        elif intent == "capability_query":
            self._build_capability_plan(plan)
        elif intent == "language_capability_query":
            self._build_language_plan(plan, question)
        elif intent == "information_query":
            self._build_definition_plan(plan, question)
        elif intent in ("explanation_request", "simplify_explanation"):
            if intent == "simplify_explanation":
                plan.style = "simple"
            self._build_explanation_plan(plan, semantic, question)
        elif intent == "research":
            self._build_research_plan(plan, semantic, question)
        elif intent == "procedural_instruction":
            self._build_procedural_plan(plan, semantic, question)
        elif intent == "suitability_query":
            self._build_suitability_plan(plan, question)
        elif intent == "analysis_request":
            self._build_analysis_plan(plan)
        elif intent == "recommendation_request":
            self._build_recommendation_plan(plan, semantic)
        elif intent == "opinion_request":
            self._build_opinion_plan(plan, semantic, question)
        elif intent == "proposal_request":
            self._build_proposal_plan(plan, semantic, question)
        elif intent == "command_query":
            self._build_command_plan(plan, question)
        elif intent == "clarification":
            self._build_clarification_plan(plan, question)
        else:
            self._build_conversation_plan(plan, semantic, question)
        # A long-form turn draws from the UNCAPPED material pool (the
        # evidence the retrieval above put on the table); every other
        # register keeps the short path.  A DOCUMENT turn draws from its
        # own retrieval: the knowledge goals' queries, validated before
        # any sentence reaches the response.
        try:
            from brain.longform import ResponseBudget
            if ResponseBudget.from_dict(plan.budget).warrants_longform():
                document_evidence = self._document_evidence(
                    structured, budget.minimum + 500)
                if document_evidence is not None:
                    plan.document_evidence = document_evidence
                    sources = [src.get("source", "") for src in
                               document_evidence.sources]
                    sources = [u for u in sources if u]
                    if sources:
                        plan.evidence = sources
                    plan.longform_material = list(
                        document_evidence.material)
                if not plan.longform_material:
                    plan.longform_material = self._longform_material(plan)
        except Exception:
            plan.longform_material = list(plan.longform_material or [])
        return plan

    def _document_evidence(self, structured, needed: int = 0):
        """The document's own retrieval: every knowledge goal addressed
        to its source, the results validated against what the goal
        asked about, synthesised into the material the response may
        use.  None when there is no document to gather for."""
        if structured is None or not structured.knowledge_goals:
            return None
        try:
            from brain.document_evidence import gather
        except Exception:
            return None
        try:
            evidence = gather(structured, needed=needed)
        except Exception:
            return None
        if not evidence.material:
            return None
        return evidence

    def _longform_material(self, plan: ResponsePlan) -> list:
        """The uncapped evidence pool for a long response: the
        concepts' full description sentences, the retrieved evidence's
        claims and the reasoning's own claims — every sentence the turn
        can honestly stand behind, deduplicated in order."""
        pool: list[str] = []
        seen: set[str] = set()

        def add(text) -> None:
            for sentence in _sentences(str(text or "")):
                key = re.sub(r"\s+", " ", sentence).strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    pool.append(sentence.strip())

        target = str(plan.target or "").strip()
        side_topics = [t.strip() for t in re.split(
            r",\s*|\s+and\s+", target, flags=re.IGNORECASE)
            if t.strip()][:5]
        if target:
            side_topics = [target] + [t for t in side_topics
                                      if t.lower() != target.lower()]
        retrieved: set[str] = set()
        for topic in side_topics:
            key = topic.lower()
            if key in retrieved:
                continue
            retrieved.add(key)
            top, related = self._retrieve_topic(topic)
            if top is not None:
                add(top.description)
                for concept in (related or [])[:3]:
                    add(getattr(concept, "description", ""))
        retrieval = plan.retrieval or {}
        for entry in (retrieval.get("evidence") or []):
            if isinstance(entry, dict):
                for claim in (entry.get("relevant_claims") or []):
                    add(claim)
        add(plan.explanation)
        for claim in plan.claims:
            add(claim)
        for related_sentence in plan.related:
            add(related_sentence)
        for step in plan.steps:
            add(step)
        for option in plan.options:
            if isinstance(option, dict):
                add(option.get("description"))
        return pool

    # ── The identity act: the self knowledge's identity, composed ──

    def _build_identity_plan(self, plan: ResponsePlan) -> None:
        """The identity answer composed from the self knowledge's
        identity items — name, version, type, philosophy — never a
        fixed introduction string."""
        identity = self.self_knowledge.get_identity() \
            if self.self_knowledge else {}
        name = str(identity.get("name", "") or plan.system_name or "RENA-BR")
        plan.system_name = name
        version = str(identity.get("version", "") or "")
        kind = str(identity.get("type", "") or "")
        philosophy = str(identity.get("philosophy", "") or "")

        if plan.requested_language == "ko":
            if plan.speech_style == "casual":
                head = (f"난 {name}이야" if _has_batchim(name)
                        else f"난 {name}야")
            else:
                head = f"저는 {name}입니다"
            if version:
                head += f" ({version})"
            if kind:
                head += f", {kind}"
            answer = head + "."
            if philosophy:
                answer += f" 원칙: {philosophy}."
        else:
            head = f"I'm {name}"
            if version:
                head += f" ({version})"
            if kind:
                head += f" — {kind.lower() if kind[:1].islower() else kind}"
            answer = head + "."
            if philosophy:
                answer += f" Philosophy: {philosophy}."
        plan.answer = answer
        plan.claims = [answer]
        plan.evidence = ["self knowledge: identity"]
        plan.confidence = 0.9

    # ── The capability query: the self knowledge's actual capabilities ──

    def _build_capability_plan(self, plan: ResponsePlan) -> None:
        caps = self.capabilities.get_active() if self.capabilities else []
        if not caps:
            caps = self.capabilities.get_implemented() \
                if self.capabilities else []
        claims = [c.description for c in caps[:6] if c.description]
        plan.claims = claims
        plan.evidence = ["capability registry: implemented capabilities"]
        plan.confidence = 0.9 if claims else 0.3

    # ── The language capability: the HONEST support check ──

    _SUPPORTED_RESPONSE_LANGUAGES = {
        "english": "yes", "영어": "yes", "korean": "yes", "한국어": "yes",
    }

    def _build_language_plan(self, plan: ResponsePlan, question: str) -> None:
        """The honest language support check: the self knowledge's actual
        languages decide — an unsupported language is reported as not
        supported, never claimed."""
        asked = None
        for lang in ("english", "korean", "japanese", "chinese",
                     "spanish", "french", "german",
                     "영어", "한국어", "일본어", "중국어"):
            if lang in question.lower():
                asked = lang
                break
        supported = None
        if asked:
            supported = self._SUPPORTED_RESPONSE_LANGUAGES.get(asked)
        if supported == "yes":
            plan.answer = ("yes" if plan.requested_language == "en"
                           else "네")
            plan.claims = ["The system responds in English and Korean."]
        elif supported is None:
            # not supported / unknown — the honest report
            plan.answer = ("not by default — English and Korean are the "
                           "supported response languages"
                           if plan.requested_language == "en"
                           else "기본적으로는 영어와 한국어를 지원합니다")
            plan.claims = ["Supported response languages: English, Korean"]
        plan.evidence = ["self knowledge: response languages"]
        plan.confidence = 0.85

    # ── The information query: the definition composed from evidence ──

    def _build_contrast_plan(self, plan: ResponsePlan, parts: list,
                             question: str = "") -> bool:
        """Compose a CONTRAST from both members' evidence.

        Each side is answered by its own evidence source — the local
        concept store when it knows the side, the external one when the
        requirement says the side needs it.  A side with neither says
        so on its own line: a contrast is never invented by filling the
        missing member with the other member's content."""
        if len(parts) != 2:
            return False
        claims: list[str] = []
        evidence: list[str] = []
        known = 0
        # Each side's STRUCTURED facets: its concept's properties and
        # the relations' concept names — the comparison's dimensions
        # are the axes both sides' knowledge actually speaks to.
        side_props: list[set] = [set(), set()]
        side_related: list[set] = [set(), set()]
        saved = (list(plan.claims), list(plan.evidence), plan.frame)
        for index, name in enumerate(parts):
            top, related_concepts = self._retrieve_topic(name)
            if top is not None:
                known += 1
                claims.append(f"{top.name}: {top.description}")
                evidence.append(f"concept: {top.id}")
                side_props[index] = set(
                    getattr(top, "properties", None) or {})
                side_related[index] = {
                    str(getattr(c, "name", "") or "").strip()
                    for c in related_concepts
                    if str(getattr(c, "name", "") or "").strip()}
                continue
            req = self._analyze_requirement(plan, question=question,
                                            target=name,
                                            local_available=False)
            side_ok = False
            if req.need_web:
                plan.claims, plan.frame = [], saved[2]
                if self._build_web_evidence(plan, name, req, query=name) \
                        and plan.claims:
                    claims.append(f"{name}: {plan.claims[0]}")
                    evidence.extend(plan.evidence[:1])
                    side_ok = True
            if not side_ok:
                claims.append(f"{name}: I found no evidence for this one")
        plan.claims, plan.evidence, plan.frame = saved
        if known == 0:
            plan.claims = []
            plan.evidence = []
            return False
        plan.claims = claims
        plan.evidence = evidence or [f"concept: {p}" for p in parts]
        plan.intent = "comparison"
        plan.confidence = 0.75
        # The comparison's structure: both subjects, and the shared
        # axes their knowledge graphs expose (an axis only counts when
        # BOTH sides carry it — an absent axis stays absent).
        shared = (side_props[0] & side_props[1]) | (
            side_related[0] & side_related[1])
        plan.comparison = {
            "subject_a": str(parts[0]).strip(),
            "subject_b": str(parts[1]).strip(),
            "dimensions": sorted(d for d in shared if d),
        }
        return True

    def _build_definition_plan(self, plan: ResponsePlan, question: str) -> None:
        """The WH-definitional question ("What is X?"): the topic's
        concept evidence reasoned into a definitional claim — the
        description is the EVIDENCE (rephrased into "X is ..."),
        never echoed.  A comparative target ("X and Y") composes a
        contrast from both concepts' evidence."""
        target = self._clean_topic(plan.target)
        # The comparative structure: two knowledge topics → a contrast
        parts = self._split_topics(target)
        if len(parts) == 2:
            if self._build_contrast_plan(plan, parts, question):
                return
        top, related = self._retrieve_topic(target)
        # The requirement (web / local / context) is decided HERE, from
        # the local probe — before any retrieval happens.
        req = self._analyze_requirement(plan, question=question,
                                        target=target,
                                        local_available=top is not None)
        # The INSTANCE ask: the local store holds the DEFINITION and the
        # turn asked for an OCCURRENCE of the topic.  The external
        # evidence is framed as the instance the request asked for; if it
        # yields nothing, the definition below follows — an example is
        # never fabricated to fill the ask.
        tried_web = False
        if plan.instance_request and req.need_web:
            tried_web = True
            if self._build_web_evidence(plan, target, req,
                                        query=str(req.query or "")):
                return
        # The requirement decides whether the EXTERNAL source is
        # consulted even when a stored definition exists: a recency
        # ask ("the latest …") is not answered by the timeless entry,
        # and the instance ask never is.
        if req.need_web and not tried_web:
            if self._build_web_evidence(plan, target, req,
                                        query=str(req.query or "")):
                return
            tried_web = True
            if str(req.freshness or "") == "recent":
                # no current evidence: the stale definition does not
                # stand in for an answer about what is current
                plan.claims = []
                plan.evidence = []
                plan.confidence = 0.2
                return
        if top is None:
            if req.need_web and not tried_web and \
                    self._build_web_evidence(plan, target, req):
                return
            plan.claims = []
            plan.evidence = []
            plan.confidence = 0.2
            return
        # The definitional composition: the evidence rephrased as a
        # definition ("X is ..."), plus the related evidence as context.
        # The claim is the evidence's SENTENCES within the register's
        # budget — the paragraph itself is never echoed.
        plan.claims = _sentences(top.description)[:_claim_budget(plan.style)]
        plan.evidence = [f"concept: {top.id}"]
        for other in related[:1]:
            if other.description:
                plan.related = _sentences(other.description)[:1]
                plan.evidence.append(f"related concept: {other.id}")
        plan.frame = self._evidence_frame(target, top.name)
        plan.confidence = 0.8

    # ── The explanation request: the target's evidence, simple style ──

    def _build_explanation_plan(self, plan: ResponsePlan, semantic,
                                question: str) -> None:
        """The explanation request: the target's concept evidence → the
        explanation composed per the style.  A conversational reference
        ("that") resolves against the conversation context's previous
        topic."""
        target = self._clean_topic(plan.target)
        # A referent that refers to nothing by itself (a demonstrative
        # or bare interrogative that the resolver did not bind) resolves
        # against the discourse — never against the retriever.
        if not target or not self._topic_contentful(target):
            target = self._previous_topic()
        plan.target = target
        top, related = self._retrieve_topic(target)
        # The requirement is decided from THIS probe, before retrieval.
        req = self._analyze_requirement(plan, question=question,
                                        target=target,
                                        local_available=top is not None)
        # The INSTANCE ask: an example is sought as an example — the
        # definition the local store holds never stands in for one.
        tried_web = False
        if plan.instance_request and req.need_web:
            tried_web = True
            if self._build_web_evidence(plan, target, req,
                                        query=str(req.query or "")):
                return
        # A recency ask is answered from the current source even when
        # the timeless entry exists — and never from the timeless
        # entry when the current source yields nothing.
        if req.need_web and not tried_web:
            if self._build_web_evidence(plan, target, req,
                                        query=str(req.query or "")):
                return
            tried_web = True
            if str(req.freshness or "") == "recent":
                plan.claims = []
                plan.evidence = []
                plan.confidence = 0.2
                return
        if top is None:
            if req.need_web and not tried_web and \
                    self._build_web_evidence(plan, target, req):
                return
            plan.claims = []
            plan.evidence = []
            plan.confidence = 0.25
            return
        plan.claims = _sentences(top.description)[:_claim_budget(plan.style)]
        plan.evidence = [f"concept: {top.id}"]
        plan.frame = self._evidence_frame(target, top.name)
        plan.confidence = 0.75

    # ── The investigation: the addressee is asked to FIND OUT ──

    def _build_research_plan(self, plan: ResponsePlan, semantic,
                             question: str) -> None:
        """The investigation: the subject the utterance asked to find
        out about, the evidence sought EXTERNALLY (it is not in the
        conversation yet) and iterated until it is sufficient, and the
        workflow the request spelled out recorded on the plan.

        The sources decide what the response may claim.  When they
        hold nothing the gap stays visible — a guess never fills it —
        while a stored concept, when there is one, is evidence like
        any other."""
        target = self._clean_topic(plan.target)
        if not target or not self._topic_contentful(target):
            # the investigation's own subject (read from the request's
            # grammar), else the discourse's topic
            scope = dict(((getattr(semantic, "dialogue_intent", None)
                           or {}).get("scope")) or {})
            target = str(scope.get("investigation_subject") or "").strip() \
                or self._previous_topic()
            plan.target = target
        top, _related = self._retrieve_topic(target)
        req = self._analyze_requirement(plan, question=question,
                                        target=target,
                                        local_available=top is not None,
                                        intent="research")
        scope = dict(((getattr(semantic, "dialogue_intent", None)
                       or {}).get("scope")) or {})
        workflow = [dict(step) for step in (scope.get("workflow") or [])
                    if isinstance(step, dict)]
        tried_web = False
        if req.need_web:
            tried_web = True
            if self._build_web_evidence(plan, target, req,
                                        query=str(req.query or "")):
                plan.research = self._research_report(plan, target,
                                                      workflow)
                return
        if top is not None:
            plan.claims = _sentences(top.description)[
                :_claim_budget(plan.style)]
            plan.evidence = [f"concept: {top.id}"]
            plan.frame = self._evidence_frame(target, top.name)
            plan.confidence = 0.6 if tried_web else 0.7
            plan.research = self._research_report(plan, target, workflow)
            return
        # Nothing to stand behind: the honest gap (the ask-back names
        # what is missing), never an invented finding.
        plan.claims = []
        plan.evidence = []
        plan.confidence = 0.2
        plan.research = self._research_report(plan, target, workflow)

    @staticmethod
    def _research_report(plan: ResponsePlan, subject: str,
                         workflow: list) -> dict:
        """What this investigation did: the subject, the stages it
        ran, the queries it issued and whether the evidence stood up —
        so the trace and the response read the same facts."""
        retrieval = dict(getattr(plan, "retrieval", None) or {})
        requirement = dict(getattr(plan, "knowledge_requirement", None)
                           or {})
        sources = [e for e in (retrieval.get("evidence") or [])
                   if isinstance(e, dict) and e.get("source")]
        return {
            "subject": str(subject or ""),
            "workflow": workflow or [],
            "queries": list(retrieval.get("queries") or []),
            "attempts": int(retrieval.get("attempts") or 0),
            "sources": len(sources),
            "need_web": bool(requirement.get("need_web")),
            "claims": len(list(plan.claims or [])),
            "sufficient": bool(plan.claims) and plan.confidence >= 0.5,
        }

    # ── The procedure: the ordered way of doing something ──

    def _build_procedural_plan(self, plan: ResponsePlan, semantic,
                               question: str) -> None:
        """The procedure request: evidence first, then the ordering —
        never a canned recipe and never the knowledge dump a
        definition would be.

        The evidence decides whether an ORDERED procedure can be
        stated (the source's own imperative sentences, in order); when
        it cannot, the same evidence is presented as the guidance it
        is, so the response stays what it knows rather than inventing
        steps it was never given."""
        target = self._clean_topic(plan.target)
        if not target or not self._topic_contentful(target):
            target = self._previous_topic()
            plan.target = target
        top, _related = self._retrieve_topic(target)
        req = self._analyze_requirement(plan, question=question,
                                        target=target,
                                        local_available=top is not None)
        if req.need_web and self._build_web_evidence(
                plan, target, req, query=str(req.query or "")):
            _order_procedure(plan)
            return
        if top is None:
            plan.claims = []
            plan.evidence = []
            plan.confidence = 0.25
            return
        plan.claims = _sentences(top.description)[:_claim_budget(plan.style)]
        plan.evidence = [f"concept: {top.id}"]
        plan.frame = self._evidence_frame(target, top.name)
        plan.confidence = 0.7
        _order_procedure(plan)

    # ── The web: external evidence, consulted on requirement ──

    def _research_context(self, topic: str, plan: ResponsePlan) -> dict:
        """The research loop's generic context: the topic, the
        resolved entity, the activated concept names and a term
        expander from the graph.  The reformulation strategies are
        transforms of THIS shape — never a canned query for a
        particular question."""
        def expand(term: str):
            try:
                hits = self._find_evidence_concepts(term)
            except Exception:
                return []
            return [str(getattr(h, "name", "") or "").strip()
                    for h in hits[:3]
                    if str(getattr(h, "name", "") or "").strip()
                    and str(getattr(h, "name", "")).lower()
                    != term.lower()]

        activated: list[str] = []
        try:
            hits = self._find_evidence_concepts(topic)
            activated = [str(getattr(h, "name", "") or "").strip()
                         for h in hits[:4]
                         if str(getattr(h, "name", "") or "").strip()]
        except Exception:
            activated = []
        request_terms: list[str] = []
        try:
            turn_semantic = getattr(self, "_turn_semantic", None)
            raw = str(getattr(turn_semantic, "raw_text", "") or "")
            request_terms = [w for w in re.findall(
                r"[A-Za-z][A-Za-z+#']{2,}|[가-힣]{2,}", raw)][:8]
        except Exception:
            request_terms = []
        return {
            "topic": str(topic or ""),
            "entity": str(getattr(plan, "target", "") or ""),
            "activated": activated,
            "request_terms": request_terms,
            "expand": expand,
        }

    @staticmethod
    def _relevant_results(results, topic, requirement,
                          question: str = "") -> list:
        """Retrieval RELEVANCE: a result that shares no content word
        with the topic or with the parts the request stated answers
        another question, and is dropped rather than quoted.  The
        honest outcome of an irrelevant web is no evidence — never a
        plausible stand-in.

        The QUESTION is part of the match too: a result whose title is
        about the past (a version history, a timeline) is dropped from
        a question that never asked for the past, when a result that is
        not remains."""
        try:
            results = list(results or [])
        except Exception:
            return []
        if not results:
            return []
        wanted = _relevance_stems(topic)
        for part in (_queries_of(requirement)):
            wanted |= _relevance_stems(part)
        wanted = {t for t in wanted if len(t) >= 5}
        if not wanted:
            return results
        kept = []
        for item in results:
            text = f"{getattr(item, 'title', '') or ''} " \
                   f"{getattr(item, 'snippet', '') or ''}"
            if wanted & _relevance_stems(text):
                kept.append(item)
        question_stems = {t for t in _relevance_stems(question)
                          if len(t) >= 5}
        if kept and question_stems \
                and not (question_stems & _PAST_STEMS):
            forward = [item for item in kept
                       if not _PAST_TITLE.search(
                           str(getattr(item, "title", "") or ""))]
            if forward:
                kept = forward
        return kept

    def _build_web_evidence(self, plan: ResponsePlan, topic: str,
                            requirement, query: str = "") -> bool:
        """Retrieve the EXTERNAL evidence for a target the requirement
        said needs it, and reason it into claims.

        `query` frames the retrieval when the request asked for
        something the topic alone does not say ("Python" asked for as
        an example is retrieved as "Python example"); the claims are
        still composed ABOUT the topic — the framing never changes what
        the evidence is evidence for.

        The retrieval result never becomes the answer: it becomes
        `RetrievedKnowledge` (evidence with provenance, ranked by
        relevance / consistency / quality / freshness), the reasoning
        derives the claims from the ranked evidence, and only those
        claims reach the response — with their attribution.
        Returns False when the web has nothing either (the caller keeps
        the honest no-knowledge response)."""
        if not topic or self.web_enabled is False:
            return False
        if not self._topic_contentful(topic):
            # A non-content topic (bare interrogative/demonstrative)
            # must never become a retrieval query: whatever comes back
            # is a knowledge dump about the wrong thing (#10).
            return False
        try:
            from brain.web_search import retrieve as _retrieve_base
            from brain.retrieved_knowledge import build_from_web
            from brain.response_representation import (
                compose_evidence_claims, frame_claims)
            from brain.research_loop import research_retrieve
        except Exception:
            return False
        retrieval_query = str(query or topic)
        question = ""
        try:
            question = str(getattr(self._turn_semantic, "raw_text",
                                   "") or "")
        except Exception:
            question = ""

        def evaluate(results, attempt_query):
            results = self._relevant_results(results, topic, requirement,
                                             question=question)
            knowledge = build_from_web(results, attempt_query,
                                       requirement, question=question)
            return knowledge if knowledge.ranked() else None

        budget = dict(getattr(plan, "budget", None) or {})
        warranted = bool(budget.get("warranted"))

        def retrieve(query: str):
            # The DEEP turn reads the article (its own ask sets how
            # much of the source may be read); a short turn keeps the
            # summary.  The evidence path downstream is the same.
            return _retrieve_base(query, deep=warranted,
                                  question=question)

        def is_sufficient(knowledge) -> bool:
            """Evidence quality for THIS budget: a short turn takes
            what stands up first; a long-form turn asks for several
            sources and a deeper claim pool before it stops."""
            try:
                claims = knowledge.claims(limit=99)
                if not claims:
                    return False
                if not warranted:
                    return True
                sources = len([e for e in knowledge.ranked()
                               if e.relevant_claims])
                return sources >= 3 and len(claims) >= 15
            except Exception:
                return False

        try:
            outcome = research_retrieve(
                retrieval_query, requirement,
                retrieve=retrieve, evaluate=evaluate,
                is_sufficient=is_sufficient,
                context=self._research_context(topic, plan),
                # a multi-part ask retrieves its parts as well as the
                # topic's name: the budget follows the request's breadth
                max_attempts=(6 if _queries_of(requirement) else 3),
                time_budget=(20.0 if _queries_of(requirement) else 10.0))
        except Exception:
            self._record_retrieval_gap(plan, retrieval_query, requirement)
            return False
        results = outcome.results
        knowledge = outcome.knowledge
        if not results or knowledge is None:
            self._record_retrieval_gap(
                plan, retrieval_query, requirement,
                attempts=getattr(outcome, "attempts", 0),
                queries=getattr(outcome, "queries", None))
            return False
        knowledge.queries = list(outcome.queries)
        knowledge.attempts = max(1, outcome.attempts)
        # the evidence is framed in the RESPONSE's language
        knowledge.language = str(getattr(plan, "requested_language", "")
                                 or "en")
        claims = compose_evidence_claims(knowledge, topic, plan.style)
        if not claims:
            self._record_retrieval_gap(
                plan, retrieval_query, requirement,
                attempts=getattr(outcome, "attempts", 0),
                queries=getattr(outcome, "queries", None),
                reason="the sources the web returned carried no claim "
                       "this request's meaning could stand on")
            return False
        plan.retrieval = knowledge.to_dict()
        # The reasoning's frame: the claim is ALREADY composed from the
        # evidence (framing included), so it is stated as it stands and
        # attributed — never re-framed over the source text.
        plan.frame = "web_summary"
        plan.claims = frame_claims(topic, claims, plan.frame)
        plan.evidence = [
            f"{e.source_type}: {e.source}" for e in knowledge.ranked()[:2]
            if e.source]
        plan.confidence = 0.6
        return True

    def _record_retrieval_gap(self, plan: ResponsePlan, query: str,
                              requirement, attempts: int = 0,
                              queries=None, reason: str = "") -> None:
        """The web was ASKED and stood up nothing this meaning could
        use: the retrieval stage still records its own attempt — the
        query it framed, how many tries it made, and the empty
        evidence — so the trace shows a retrieval that FAILED rather
        than one that never ran, and the response's shortfall is
        attributable to a stage instead of vanishing with it."""
        plan.retrieval = {
            "query": str(query or ""),
            "queries": [str(q) for q in (queries or []) if q],
            "attempts": int(attempts or 0),
            "evidence": [],
            "web_used": False,
            "language": str(getattr(plan, "requested_language", "")
                            or "en"),
            "reason": str(
                reason or "the web was asked and stood up no evidence "
                          "this request could use"),
        }

    def _web_fallback(self, plan: ResponsePlan, topic: str) -> bool:
        """Backward-compatible entry: the web consulted under the
        requirement analysis (the same evidence → reasoning path)."""
        req = self._analyze_requirement(plan, question="", target=topic,
                                        local_available=False)
        return req.need_web and self._build_web_evidence(plan, topic, req)

    def _build_suitability_plan(self, plan: ResponsePlan,
                                question: str) -> None:
        """A polar question about a concept's property ("Is Python
        suitable for web development?"): reason from the evidence —
        the concept's description mentions the property or not — and
        answer yes/no with the evidence."""
        target = self._clean_topic(plan.target)
        top, related = self._retrieve_topic(target)
        if top is None:
            plan.claims = []
            plan.confidence = 0.25
            return
        # The property asked: the question's predicate ("web
        # development", "data analysis") — match against the evidence
        desc = top.description.lower()
        asked_property = self._extract_property(question)
        if asked_property and asked_property in desc:
            plan.answer = ("yes" if plan.requested_language == "en"
                           else "네")
            plan.claims = [top.description]
        elif asked_property:
            plan.answer = ("the evidence does not mention that use "
                           if plan.requested_language == "en"
                           else "그 용도에 대한 근거가 없습니다")
            plan.claims = [top.description]
        else:
            plan.claims = [top.description]
        plan.evidence = [f"concept: {top.id}"]
        plan.confidence = 0.7

    @staticmethod
    def _extract_property(question: str) -> str:
        """The property predicate of a suitability question: the
        prepositional phrase ("suitable for X", "good for X",
        "~에 적합") — the asked use."""
        m = re.search(r"(?:suitable|good|used|appropriate|realistic|"
                      r"feasible|possible|reasonable|effective|right)\s+"
                      r"for\s+(.+?)[?.\s]*$", question.lower())
        if m:
            return m.group(1).strip()
        m = re.search(r"(.+?)에\s+(?:적합|좋|유용)", question)
        if m:
            return m.group(1).strip()
        return ""

    # ── The analysis request: the honest ask-for-the-code routing ──

    def _build_analysis_plan(self, plan: ResponsePlan) -> None:
        """A polar question about a program's behavior ("Does this
        program work?"): the program is not provided in the question —
        the honest routing asks for the code to analyze."""
        if plan.requested_language == "ko":
            plan.answer = ("분석할 프로그램 코드가 아직 제공되지 않았습니다. "
                           "코드를 붙여넣어 주시면 동작을 분석해드리겠습니다.")
        else:
            plan.answer = ("No program was provided to analyze. "
                           "Paste the code and I'll analyze how it works.")
        plan.evidence = ["analysis requires the code artifact"]
        plan.confidence = 0.6

    # ── The recommendation act ──

    # The knowledge categories a recommendation may DRAW OPTIONS from.
    # A data operation or a shell command is a component of an answer,
    # not a direction a person can choose — so they are excluded.
    _OPTION_CATEGORIES = (
        "language_fundamental", "language_feature", "programming_language",
        "capability", "pattern", "algorithmic_primitive",
    )
    # The topic's own concept may only ANCHOR the options when it is one
    # of these: things a person chooses BETWEEN.  A topic that merely
    # resolves to a concept ("programming" → Software Project) restates
    # the question instead of answering it.
    _CHOICE_CATEGORIES = (
        "language_fundamental", "language_feature", "programming_language",
        "pattern",
    )

    def _build_recommendation_plan(self, plan: ResponsePlan,
                                   semantic,
                                   anchored_only: bool = False) -> None:
        """The recommendation act: the knowledge graph's candidate
        options and their dependency structure, composed into a
        recommendation.

        The OPTIONS come from the knowledge base's choice categories —
        first anchored by the topic (only when the topic resolves to
        something a person chooses between), then by the categories this
        utterance actually activated, then by the previous turn's topic.
        The ORDER comes from the graph's prerequisite relations.  When
        no option set exists at all, the act is what it then honestly
        is: a request for the referent it is missing.

        `anchored_only` is the PROPOSAL's reading: a candidate that was
        ASKED FOR must be anchored by the turn's own subject or by the
        conversation's.  An option the content's categories merely happen
        to contain answers a different question, so it is never offered
        as the candidate that was asked for.
        """
        topics = list(getattr(semantic, "intent_topics", None) or [])
        if anchored_only:
            # the proposal's own subject decides the anchor: the
            # speaker's state that prompted it is not the topic
            topics = self._split_topics(
                self._clean_topic(plan.target)) or topics
        if not topics:
            topics = self._split_topics(self._clean_topic(plan.target))
        claims: list[str] = []
        evidence: list[str] = []
        seen: set[str] = set()

        options: list[dict] = []

        def add(concept, prefix: str = "") -> None:
            if concept is None or concept.id in seen:
                return
            if not concept.description:
                return
            seen.add(concept.id)
            claims.append(f"{prefix}{concept.name}: {concept.description}")
            evidence.append(f"concept: {concept.id}")
            options.append({"name": concept.name,
                            "category": concept.category,
                            "description": concept.description})

        # 1. The topic anchors the options only when it resolves to a
        #    CHOICE — with its prerequisites first, the graph's
        #    dependency structure deciding the order
        for t in topics[:2]:
            top, related = self._retrieve_topic(t)
            if top is None or top.category not in self._CHOICE_CATEGORIES:
                continue
            if self.relations:
                for rel in self.relations.find_related(top.id):
                    if rel.relation_type.value != "requires":
                        continue
                    other_id = rel.target_id if rel.source_id == top.id \
                        else rel.source_id
                    other = self.concepts.get(other_id) if other_id != top.id                         else None
                    if other is not None \
                            and other.category in self._OPTION_CATEGORIES:
                        add(other)
            add(top)
            for r in related[:2]:
                if r.category in self._OPTION_CATEGORIES:
                    add(r)

        # 2. The content's own knowledge domains — the categories this
        #    utterance activated (never a topic keyword)
        if not claims and not anchored_only:
            for cat in self._ordered_option_categories(
                    self._activated_categories(semantic)):
                for c in self.concepts.list_by_category(cat)[:4]:
                    add(c)
                if len(claims) >= 4:
                    break

        # 3. The conversation's referent: the previous turn's topic,
        #    when this turn named none of its own.  A DIRECTION a person
        #    chooses between counts; a referent the discourse merely
        #    resolved to (an artifact, a technique) restates the
        #    question — it is not an option.
        implied_foundations = False
        if not claims:
            for prev in self._split_topics(self._previous_topic()):
                top, related = self._retrieve_topic(prev)
                if top is None:
                    continue
                if top.category not in self._CHOICE_CATEGORIES:
                    implied_foundations = True
                    continue
                add(top)
                for r in related[:2]:
                    if r.category in self._OPTION_CATEGORIES:
                        add(r)
                if claims:
                    break

        # 4. The direction a non-choice referent IMPLIES: the discourse
        #    resolved to something that is BUILT FROM the fundamentals
        #    ("learning programming" → a software project) — the
        #    direction it implies is its foundation, in prerequisite
        #    order, composed from the knowledge base's own concepts.
        if not claims and implied_foundations and not anchored_only:
            for cat in self._ordered_option_categories(
                    ("language_fundamental", "language_feature")):
                for c in self.concepts.list_by_category(cat)[:4]:
                    add(c)
                if len(claims) >= 4:
                    break

        if not claims:
            # No option set exists — the recommendation's missing
            # referent decides what is asked back.
            self._build_clarification_plan(plan, plan.target or
                                           self._previous_topic())
            plan.intent = "clarification"
            return
        plan.claims = claims[:4]
        plan.evidence = evidence[:6]
        plan.options = options[:4]
        plan.confidence = min(0.8, 0.35 + 0.12 * len(plan.claims))

    # ── The stance act: the turn's own claim is the subject ──

    @staticmethod
    def _proposition_of(question: str) -> str:
        """The clause the turn STATES — the claim the speaker holds and
        the reply discusses.  The declarative sentence beside the stance
        question carries it; when the claim is only the question's own
        complement, that complement is it.  Never the question's frame."""
        text = str(question or "").strip()
        if not text:
            return ""
        best = ""
        for clause in re.split(r"(?<=[.!?？。])\s+", text):
            piece = clause.strip()
            if not piece or "?" in piece or "？" in piece:
                continue
            if len(piece.split()) >= len(best.split()):
                best = piece
        if best:
            return best
        complement = re.search(
            r"\b(?:think|believe|agree|feel|reckon)\s+(?:that\s+)?"
            r"([^?!？。]{3,}?)[.?!？。]*\s*$",
            text, re.IGNORECASE)
        if complement and len(complement.group(1).split()) >= 3:
            return complement.group(1).strip()
        return text

    def _build_opinion_plan(self, plan: ResponsePlan, semantic,
                            question: str) -> None:
        """The stance act: the proposition the speaker HOLDS is what is
        discussed, so the knowledge is sought for the claim's own topic
        and the evidence speaks to it — never a definition of the topic
        the claim mentions, never a bare agreement.

        The evidence may back the claim or undercut it: the frame marks
        it as SUPPORT, so the reply states what the evidence says about
        the claim instead of restating the claim as a fact."""
        proposition = self._proposition_of(question)
        plan.proposition = proposition
        topics = list(getattr(semantic, "intent_topics", None) or [])
        if not topics:
            from brain.dialogue_act import extract_topics as _topics_of
            topics = [t for t in (_topics_of(proposition or question) or [])
                      if t]
        if topics:
            plan.target = str(topics[0])
        plan.frame = "clause"
        top, _related = self._retrieve_topic(plan.target)
        req = self._analyze_requirement(
            plan, semantic=semantic, question=question,
            target=plan.target, local_available=top is not None,
            intent=plan.intent)
        if req.need_web and self._build_web_evidence(
                plan, plan.target, req, query=str(req.query or "")):
            plan.confidence = min(plan.confidence or 0.7, 0.7)
            return
        if top is None:
            plan.claims = []
            plan.evidence = []
            plan.confidence = 0.3
            return
        plan.claims = _sentences(top.description)[:_claim_budget(plan.style)]
        plan.evidence = [f"concept: {top.id}"]
        plan.confidence = 0.7

    def _build_proposal_plan(self, plan: ResponsePlan, semantic,
                             question: str) -> None:
        """The proposal act: the turn asked for a CANDIDATE, so the
        knowledge graph's own option structure is what is offered — the
        same anchoring the recommendation act uses (the topic's choice
        category, then the conversation's referent), held to the
        proposal's stricter reading: the candidate must be anchored by
        the turn's own subject.

        The SUBJECT is the candidate the request asked to be given, not
        the speaker's state that asked for it — "I'm bored. Suggest
        something fun" is about the something-fun, never about the
        boredom.

        When no option structure exists at all, the act is honestly the
        question it then is: what kind of candidate is being asked for.
        The builder never invents options the knowledge does not hold."""
        target = (plan.target or "").strip()
        if not target or re.match(
                r"^(?:i|i['’]?m|i\s+am|i['’]?ve|it|저는|나는|제가)\b",
                target, re.IGNORECASE):
            from brain.dialogue_act import candidate_subject
            subject = candidate_subject(question)
            if subject:
                plan.target = subject
        self._build_recommendation_plan(plan, semantic, anchored_only=True)
        if plan.intent == "clarification":
            # Nothing anchors the candidate, and there is no referring
            # expression to resolve either — what is missing IS the
            # subject.  The ask-back says that, rather than asking
            # which part of a candidate the speaker never referred to.
            plan.answer = plan.target
            plan.reference = {}
            plan.target = ""
            return
        plan.confidence = min(plan.confidence or 0.7, 0.75)

    def _ordered_option_categories(self, cats) -> list:
        """The activated categories restricted to the option categories,
        in prerequisite order: foundations before features, features
        before the languages and artifacts built from them."""
        rank = {c: i for i, c in enumerate(self._OPTION_CATEGORIES)}
        picked = [c for c in cats if c in rank]
        picked.sort(key=lambda c: rank[c])
        return picked

    def _activated_categories(self, semantic) -> list:
        """The categories of the concepts this utterance activated —
        the content's own knowledge domains."""
        cats: list[str] = []
        for cid in (getattr(semantic, "activated_concept_ids", None) or []):
            c = self.concepts.get(cid) if self.concepts else None
            if c is None:
                continue
            if c.category.startswith("shell"):
                continue
            if c.category not in cats:
                cats.append(c.category)
        return cats

    # ── The command query: a composed how-to, never a knowledge dump ──

    def _build_command_plan(self, plan: ResponsePlan, question: str) -> None:
        """The command query: ONE shell operation — the one whose tags
        this question actually names and that yields a real approach in
        the detected shell — composed with the environment it runs in.

        Ranking is content-derived, never a stored order: an operation
        is chosen because its tags match this question's words and it
        has an actionable approach here.
        """
        try:
            from brain.shell_knowledge import (
                reason_about_shell, _gather_operation_context,
            )
            from brain.concept_store import ConceptStore
        except Exception:
            plan.claims = []
            plan.confidence = 0.15
            return
        detected = reason_about_shell(question, self.concepts, self.relations)
        op_ids = list(detected.get("operations") or [])
        if not op_ids:
            plan.claims = []
            plan.evidence = []
            plan.confidence = 0.2
            return

        shell_id = detected.get("shell")
        words = {w for w in ConceptStore._tokenize_query(question)}
        ranked: list[tuple[int, int, str]] = []
        for op_id in op_ids:
            ctx = _gather_operation_context(op_id, self.concepts, self.relations)
            concept = self.concepts.get(op_id)
            hits = 0
            if concept:
                for tag in concept.tags:
                    tag_lower = tag.lower()
                    for w in words:
                        if len(w) >= 3 and ConceptStore._word_in(tag_lower, w):
                            hits += 1
                            break
            shell_concept = self.concepts.get(shell_id) if shell_id else None
            shell_name = (shell_concept.name if shell_concept else shell_id)
            has_shell_approach = bool(
                shell_name and ctx["approaches"].get(shell_name))
            # An operation that actually runs here, that this question
            # names, is the answer; a described-but-unapproachable
            # concept is only a definition.
            ranked.append((1 if has_shell_approach else 0, hits, op_id))
        ranked.sort(key=lambda x: (-x[0], -x[1]))
        best_approach_flag, _, op_id = ranked[0]
        concept = self.concepts.get(op_id)
        ctx = _gather_operation_context(op_id, self.concepts, self.relations)

        claims: list[str] = []
        evidence: list[str] = []
        if ctx["description"]:
            claims.append(ctx["description"])
        approaches = ctx["approaches"]
        shell_concept = self.concepts.get(shell_id) if shell_id else None
        shell_name = shell_concept.name if shell_concept else shell_id
        if shell_name and approaches.get(shell_name):
            claims.append(f"{shell_name}: {approaches[shell_name]}")
        elif approaches:
            shell_name, approach = next(iter(approaches.items()))
            claims.append(f"{shell_name}: {approach}")
        if concept:
            evidence.append(f"concept: {concept.id}")
        if shell_id:
            evidence.append(f"shell: {shell_id}")
        if detected.get("platform"):
            evidence.append(f"platform: {detected['platform']}")
        plan.claims = claims
        plan.evidence = evidence
        plan.intent = "command_query"
        plan.confidence = min(0.8, 0.4 + 0.15 * len(claims))

    # ── The clarification act ──

    def _build_clarification_plan(self, plan: ResponsePlan,
                                  question: str) -> None:
        """The clarification act: the missing structure decides what is
        asked back — the referent the parse could not resolve."""
        target = self._clean_topic(plan.target)
        plan.answer = target
        plan.claims = []
        plan.options = []
        plan.evidence = ["clarification act: referent unresolved"]
        plan.confidence = 0.3

    # ── The conversation act ──

    def _build_conversation_plan(self, plan: ResponsePlan, semantic,
                                 question: str) -> None:
        """The conversation act: an acknowledgment composed from the
        utterance's recognized structure (the act's predicate and the
        topic) plus a grounded offer of what the system can actually do
        — never a knowledge dump and never a fixed reply."""
        comps = getattr(semantic, 'composition', {}) or {}
        cues = [c.get("primitive", "") for c in (comps.get("user_cues") or [])]
        act = getattr(semantic, "dialogue_act", None) or {}
        plan.answer = (plan.target or "").strip()
        plan.claims = []
        plan.evidence = ["conversation act",
                         f"force={act.get('force', '')}",
                         f"predicate={act.get('predicate_class', '')}"] \
            + cues[:2]
        plan.confidence = 0.55

    # ------------------------------------------------------------------
    # Topic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_topic(target: str) -> str:
        """Clean the topic reference: the leading articles ("a", "an",
        "the") and the trailing question mark are grammar, not content.
            "a pointer" → "pointer"
            "an average" → "average"
        """
        t = target.strip().rstrip("?？.").strip()
        t = re.sub(r"^(?:please\s+)?(?:explain|describe|tell\s+me\s+about)\s+",
                   "", t, flags=re.IGNORECASE).strip()
        t = re.sub(
            r"[\s,;:-]*(?:in\s+(?:exactly\s+)?(?:\d+|one|two|three|four|"
            r"five)\s+sentences?|in\s+plain\s+language|"
            r"in\s+simple\s+terms|simply|briefly|concisely|"
            r"한\s*문장으로|간단히|쉽게)\s*$",
            "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"^(a|an|the)\s+", "", t, flags=re.IGNORECASE).strip()
        return t

    # The Korean interrogative predicates: a closed grammatical class
    # (like the WH-words in _strip_interrogative) — never a topic
    # vocabulary.
    _KR_INTERROGATIVE = (
        r"(?:뭐야|뭐예요|뭐니|무엇(?:인가(?:요)?|일까|야)?|"
        r"어때(?:요)?|왜|누구(?:야|예요)?|언제|어디|어떻게|"
        r"얼마(?:인가(?:요)?)?|어떤)"
    )

    @staticmethod
    def _topic_from_question(text: str) -> str:
        """The content topic under the question's own framing.

        The interrogative clause is GRAMMAR, not content — stripping it
        yields what the question is about:
            "Python이 뭐야" → "Python"
            "What is a compiler?" → "a compiler"
            "그거 어때?" → "그거"
        A bare interrogative ("왜?") has no content topic and comes
        back unchanged — _topic_contentful decides what that means.
        """
        t = ConversationKnowledgeEngine._strip_interrogative(text)
        m = re.match(
            r"^(?P<topic>.+?)\s*(?:이|가|은|는)?\s*"
            + ConversationKnowledgeEngine._KR_INTERROGATIVE +
            r"(?:(?:이|가)?(?:요|까|할까|까요))?\s*[?？]?\s*$",
            t.strip())
        if m:
            return m.group("topic").strip()
        return t

    @staticmethod
    def _content_target(target: str, structured) -> str:
        """The part of the discourse target the document read as
        content.

        A segment found only in a response shaping clause answers how
        the reply must be written, not what it is about; it is dropped
        so the subject never carries the reply's own framing.  A
        segment the content clauses also carry stays: the speaker named
        it in what they asked."""
        text = str(target or "").strip()
        if not text or structured is None:
            return text
        clauses = list(getattr(structured, "clauses", None) or [])
        shaping = [str(c.get("text") or "").lower() for c in clauses
                   if str(c.get("kind") or "") == "response_shaping"]
        if not shaping:
            return text
        content = [str(c.get("text") or "").lower() for c in clauses
                   if str(c.get("kind") or "")
                   in ("knowledge", "reply_subject")]
        parts = [p.strip() for p in re.split(r"\s+and\s+", text) if p.strip()]
        kept = [part for part in parts
                if not (any(part.lower() in clause for clause in shaping)
                        and not any(part.lower() in clause
                                     for clause in content))]
        if not kept:
            return ""
        return " and ".join(kept)

    @staticmethod
    def _topic_contentful(topic: str) -> bool:
        """Whether a target actually REFERS to something.

        The bare interrogatives and the demonstratives without an
        antecedent are language, not knowledge targets: retrieving
        with them as queries matches whatever the store (or the web)
        happens to contain — a knowledge dump about the wrong thing.

        The request's OWN clause is equally not a target: a
        performative head over a clausal object ("explain that more
        simply", "tell me about space") is the speaker's ASK, not what
        they ask about.  A retrieval with it as a query answers about
        the ask's own words instead of its referent."""
        t = (topic or "").strip().lower().rstrip("??.!")
        if not t:
            return False
        if t in ConversationKnowledgeEngine._BARE_INTERROGATIVES \
                or t in ConversationKnowledgeEngine._DEICTICS:
            return False
        return not ConversationKnowledgeEngine._is_request_clause(t)

    # The request's clause: a performative/auxiliary head (the speaker
    # ASKING) together with a clause marker — an imperative over its
    # object.  A bare noun phrase that happens to start with one of
    # these words ("list operations") is a topic, not a request, and
    # stays a target: the marker is what makes it a clause.
    _REQUEST_CLAUSE_HEAD = re.compile(
        r"^(?:please\s+|plz\s+|pls\s+|let'?s\s+|can you\s+|could you\s+|"
        r"would you\s+|will you\s+|tell me\s+|give me\s+|show me\s+|"
        r"help me\s+|explain\b|describe\b|forget\b|remember\b|stop\b|"
        r"continue\b|start\b|repeat\b|define\b|discuss\b|"
        r"talk about\b|mention\b|say\b|make\b|create\b|build\b|"
        r"write\b|generate\b|implement\b|design\b|list\b|find\b|"
        r"check\b|compare\b|print\b|return\b|remove\b|filter\b|"
        r"sort\b|설명|알려|가르쳐|잊어|만들|작성|찾|보여|출력|정렬)"
        r"(?:\s|\b)", re.IGNORECASE)
    _REQUEST_CLAUSE_MARKER = re.compile(
        r"\b(?:me|you|it|that|this|these|those|us|them|him|her|"
        r"something|anything|everything|can|could|would|will|whether|"
        r"how)\b|"
        r"(?:그거|그것|이거|이것|저거|좀|해줘|줄래)", re.IGNORECASE)

    @classmethod
    def _is_request_clause(cls, text: str) -> bool:
        """The text is the request's own clause: a performative head
        with a clausal object.  Structural — the head class and the
        marker decide, never a sentence→answer map."""
        t = (text or "").strip()
        if len(t.split()) < 2:
            # a single word is a term, not a clause
            return False
        if not cls._REQUEST_CLAUSE_HEAD.match(t):
            return False
        return bool(cls._REQUEST_CLAUSE_MARKER.search(t))

    # The instance-request framing: the noun class of an ask for an
    # OCCURRENCE of the topic (an example, a sample, a case).  The noun
    # plus the act's own form (an ask, never an assertion) decide —
    # never a sentence→answer map.
    _INSTANCE_NOUNS = re.compile(
        r"\b(?:example|instance|sample|case)s?\b|예시|예제|사례|보기",
        re.IGNORECASE)

    def _instance_request(self, question: str, semantic) -> str:
        """The INSTANCE noun this turn asks for, or "" when it does not
        ask for one.  The framing is the speaker's own word, kept so the
        retrieval can be framed the way the request framed it."""
        text = str(question or "")
        match = self._INSTANCE_NOUNS.search(text)
        if not match:
            return ""
        act = dict(getattr(semantic, "dialogue_act", None) or {})
        form = str(act.get("form") or "").lower()
        force = str(act.get("force") or "").lower()
        if form in ("imperative", "interrogative") or force == "request":
            return match.group(0).lower()
        return ""

    # Closed grammatical classes — language, never knowledge targets.
    _BARE_INTERROGATIVES = {
        "왜", "언제", "어디", "누구", "뭐", "무엇", "어떻게", "얼마",
        "how", "why", "when", "where", "who", "what", "which",
    }
    _DEICTICS = {
        "그거", "이거", "그것", "이것", "저거",
        "that", "this", "it", "those", "these",
    }

    @staticmethod
    def _evidence_frame(topic: str, name: str) -> str:
        """How the evidence relates to the topic.

        "definition" when the topic IS the evidence's concept — the
        claim states what the topic is.  "clause" when the topic is the
        question's own proposition ("programming languages need
        different types") and the evidence merely SUPPORTS it: the
        claim then follows the proposition as its content, never
        rewrites it as "X is ..." (which would be nonsense for a
        sentence).
        """
        t = ConversationKnowledgeEngine._clean_topic(topic).lower()
        n = (name or "").strip().lower()
        if not t or not n:
            return "definition"
        if t == n or t in (f"a {n}", f"an {n}", f"the {n}"):
            return "definition"
        if len(t.split()) <= 3 and (n in t or t in n):
            return "definition"
        return "clause"

    @staticmethod
    def _split_topics(target: str) -> list:
        """The comparative structure: "X and Y" → [X, Y] (a comparison
        target names two topics)."""
        parts = re.split(r"\s+(?:and|vs\.?|versus|와|과|이랑|랑)\s+",
                         target, flags=re.IGNORECASE)
        return [p.strip() for p in parts
                if p.strip() and len(p.strip()) >= 2][:2]

    def _previous_topic(self) -> str:
        """The conversation context's previous topic (the reference
        resolution's source: "that" refers to it).  The topic state's
        stable referent (a reset retires it) is the primary source;
        without one the stored turn is the raw utterance — the
        interrogative framing is stripped so the reference becomes a
        TOPIC ("What is a compiler?" → "a compiler"), not a nested
        question ("What is a compiler is...")."""
        provider = getattr(self, "topic_provider", None)
        if provider is not None:
            try:
                stable = provider.current_topic
            except Exception:
                stable = ""
            if stable:
                return stable
        for turn in reversed(self.conversation_history[-3:]):
            user_input = getattr(turn, 'user_input', '') \
                if hasattr(turn, 'user_input') else \
                (turn.get("user_input", "") if hasattr(turn, 'get') else "")
            if not user_input:
                continue
            # A bare expressive (thanks / goodbye) carries no topic —
            # the reference resolution skips it to the previous
            # topical turn.
            bare = user_input.strip().lower().rstrip("!?.~")
            if any(bare == form or bare.startswith(form + " ")
                   for _act, forms in self._EXPRESSIVE_ACTS
                   for form in forms):
                continue
            return self._strip_interrogative(user_input[:60])
        return ""

    @staticmethod
    def _strip_interrogative(text: str) -> str:
        """The grammatical WH-front + copula/auxiliary framing removed:
        the topic is what remains ("What is X?" → "X")."""
        t = text.strip().rstrip("??.!")
        t = re.sub(
            r"^(what|who|whom|which|how|why|when|where)"
            r"(?:\s*'(?:s|re)\b|\s+(?:(?:is|are|was|were|do|does|did|"
            r"can|could|would|will|should|to)\b))?\s*",
            "", t, flags=re.IGNORECASE)
        return t.strip() or text.strip()


# ============================================================
# Response realization (the ResponsePlan → the natural language)
# ============================================================

def _realize_social(plan: ResponsePlan, intent: str, lang: str) -> str:
    """The social reply composed from the act + the identity + the
    offer — each slot realizes from its variants at generation time
    (the ResponseVariator: two runs of the same meaning surface
    differently; the identity always carries the self knowledge's
    name)."""
    name = plan.system_name or "RENA-BR"
    casual = plan.speech_style == "casual"
    if lang == "ko":
        name_p = "이야" if _has_batchim(name) else "야"
        if intent in ("greeting", "salutation"):
            if casual:
                opener = choice(["안녕!", "반가워!"])
                ident = choice([f"난 {name}{name_p}.", f"{name}{name_p}."])
                offer = choice(["뭐 도와줄까?", "무엇을 만들까?",
                                "오늘 필요한 건 뭐야?"])
            else:
                opener = choice(["안녕하세요!", "안녕하세요.", "반가워요!"])
                ident = choice([f"저는 {name}입니다.", f"{name}입니다."])
                offer = choice(["무엇을 도와드릴까요?", "어떤 것을 만들까요?",
                                "오늘 무엇이 필요하신가요?"])
            return f"{opener} {ident} {offer}"
        if intent == "gratitude":
            if casual:
                opener = choice(["천만에야!", "아니야, 별말이야."])
                tail = choice(["또 필요하면 말해.",
                               "궁금한 거 있으면 물어봐.",
                               "언제든 편하게 불러."])
            else:
                opener = choice(["천만에요!", "별말씀을요.",
                                 "도움이 되어 기쁩니다."])
                tail = choice(["또 필요하시면 말씀해 주세요.",
                               "다른 것도 궁금한 점이 있으면 물어보세요.",
                               "언제든 편하게 불러주세요."])
            return f"{opener} {tail}"
        if intent == "valediction":
            if casual:
                opener = choice(["안녕!", "다음에 보자.", "잠시 후에 보자."])
                tail = choice([f"{name}{name_p} — 언제든 다시 불러.",
                               f"{name}{name_p} — 또 보자."])
            else:
                opener = choice(["안녕히 가세요!", "다음에 또 봬요.",
                                 "잠시 후에 또 뵐게요."])
                # the name stands alone (no particle) — English names do
                # not take Korean josa cleanly
                tail = choice([f"저는 {name}입니다 — 언제든 다시 불러주세요.",
                               f"{name}입니다 — 또 만날게요."])
            return f"{opener} {tail}"
        if intent == "state_inquiry":
            # honest: no mood — system state, then back to the user
            if casual:
                opener = choice(["난 감정은 없지만 상태는 좋아.",
                                 "기분 같은 건 없어, 시스템은 잘 돌아가고 있어."])
                tail = choice(["너는 어때?", "오늘은 뭐 하려고 해?",
                               "뭐 도와줄까?"])
            else:
                opener = choice(["저는 감정이 없어서 기분은 없지만, "
                                 "시스템은 정상 작동 중입니다.",
                                 "기분을 느끼지는 못하지만 상태는 양호합니다."])
                tail = choice(["고객님은 어떠신가요?",
                               "오늘은 무엇을 도와드릴까요?",
                               "무엇이 필요하신가요?"])
            return f"{opener} {tail}"
        if casual:
            return choice(["응, 더 이야기해도 좋고 뭐든 물어봐.",
                           "물론이야 — 더 말해봐.",
                           "좋아, 어떤 이야기 이어갈까?"])
        return choice(["네, 더 이야기해주시거나 무엇이든 물어보세요.",
                       "물론이죠 — 더 말씀해 주세요.",
                       "좋아요, 어떤 이야기를 이어갈까요?"])
    if intent in ("greeting", "salutation"):
        opener = choice(["Hello!", "Hi!", "Hey!"])
        ident = choice([f"I'm {name}.", f"{name} here."])
        offer = choice(["How can I help you today?",
                        "What would you like to work on?",
                        "What are we building today?"])
        return f"{opener} {ident} {offer}"
    if intent == "gratitude":
        opener = choice(["You're welcome!", "Happy to help!", "Anytime!"])
        tail = choice(["Ask me anything else whenever you need.",
                       "Let me know if there's anything else.",
                       "What else can I do for you?"])
        return f"{opener} {tail}"
    if intent == "valediction":
        opener = choice(["Goodbye!", "See you next time!", "Take care!"])
        tail = choice([f"Come back any time — I'm {name}.",
                       f"{name} here whenever you need me."])
        return f"{opener} {tail}"
    if intent == "state_inquiry":
        # honest: no mood — system state, then back to the user
        opener = choice([
            "I don't have moods, but the system is running fine.",
            "No feelings here — everything is healthy on my side."])
        tail = choice(["How are you doing?",
                       "What are you working on today?",
                       "What can I help you with?"])
        return f"{opener} {tail}"
    # The conversation act: an acknowledgment composed from the
    # recognized content plus a grounded follow-up — never a fixed
    # reply, never a knowledge dump.
    return _realize_conversation(plan, "en" if lang != "ko" else "ko")


def _realize_conversation(plan: ResponsePlan, lang: str) -> str:
    """The conversation act: an acknowledgment composed from the
    recognized content (the topic the parse resolved and the predicate
    the act layer classified) plus a grounded follow-up — never a fixed
    reply, never a knowledge dump."""
    topic = (plan.answer or plan.target or "").strip()
    predicate = ""
    for ev in plan.evidence:
        if str(ev).startswith("predicate="):
            predicate = str(ev).split("=", 1)[1]
    casual = plan.speech_style == "casual"
    if re.search(r"(frustrat|stuck|annoy|upset|angry|overwhelm|"
                 r"disappoint|답답|짜증|속상|힘들)", topic,
                 re.IGNORECASE):
        if lang == "ko":
            return ("답답하시겠어요. 실패하는 테스트나 오류 메시지를 "
                    "보여주시면 어디서 막히는지 함께 좁혀보겠습니다.")
        return ("That sounds frustrating. Share the failing test or the "
                "error message and I will help narrow down what is going "
                "wrong.")
    if lang == "ko":
        if predicate == "advice":
            head = (f"{topic} 쪽으로 고민 중이시군요." if topic
                    else "고민이 있으시군요.")
        elif predicate == "evaluation":
            head = (f"{topic}의 가능성을 살피고 계시는군요." if topic
                    else "가능성을 살피고 계시는군요.")
        else:
            head = f"{topic}에 대해 말씀하시는군요." if topic                 else "말씀 고맙습니다."
        tail = choice([
            "어떻게 이어갈지 알려주시면 도와드리겠습니다.",
            "설명이 필요하면 말해주세요.",
        ]) if not casual else choice([
            "어떻게 이어갈지 말해줘.",
            "설명이 필요하면 말해줘.",
        ])
        return f"{head} {tail}"
    if predicate == "advice":
        head = f"I follow — you're weighing up {topic}." if topic             else "I follow — you're thinking it through."
    elif predicate == "evaluation":
        head = f"I follow — you're sizing up {topic}." if topic             else "I follow — you're sizing it up."
    else:
        head = f"I follow — you mentioned {topic}." if topic             else "I follow."
    tail = choice([
        "Where would you like to take it from here?",
        "Tell me the next move and I'll follow along.",
    ])
    return f"{head} {tail}"


def _no_evidence_reply(plan: ResponsePlan, lang: str) -> str:
    """The reply when NEITHER the local store nor the web produced
    evidence.

    It states what was sought, where it was sought and why that
    settled nothing — from the requirement and the retrieval the turn
    actually ran — and then asks along the axis the turn is actually
    on (its intent, its freshness, its instance ask).  No angle list,
    no canned "couldn't find it": a fixed string would say the same
    thing about every turn and therefore nothing about this one."""
    req = dict(plan.knowledge_requirement or {})
    target = (plan.target or "").strip().rstrip("??.!")
    query = str(req.get("query") or "").strip() or target
    need_web = bool(req.get("need_web"))
    scope = str(req.get("time_scope") or "").strip()
    fresh = str(req.get("freshness") or "").strip() in ("recent", "current")
    ko = str(lang or plan.requested_language or "").startswith("ko")
    casual = plan.speech_style == "casual"

    # What is missing: the axis the next turn has to narrow along,
    # decided by the request's own structure.
    if plan.instance_request:
        axis = ("어떤 종류의 예시인지" if ko else "the kind of example")
    elif plan.intent == "procedural_instruction":
        axis = ("어느 단계/상황인지" if ko else "which step you are on")
    elif need_web and (fresh or scope):
        axis = ("대략 언제인지" if ko else "roughly when")
    elif not target:
        axis = ("더 구체적인 주제" if ko else "a more specific topic")
    else:
        axis = ("어느 쪽인지" if ko else "the part you need")

    if ko:
        head = (f"'{target}'에 대한 근거를 아직 찾지 못했어." if casual
                else f"'{target}'에 대한 근거를 아직 찾지 못했습니다.") \
            if target else ("아직 근거를 찾지 못했어." if casual
                            else "아직 근거를 찾지 못했습니다.")
        if need_web:
            looked = f"'{query}'로 검색" + (f"({scope})" if scope else "")
            head += (f" {looked}을 했는데 검증할 만한 게 나지 않았어."
                     if casual else
                     f" {looked}을 했지만 검증할 수 있는 결과가 없었습니다.")
        else:
            head += (" 내가 아는 범위에도 없어." if casual else
                     " 제 지식 범위에도 없습니다.")
        head += (f" {axis}만 알려주면 다시 찾아볼게." if casual else
                 f" {axis}을 알려주시면 다시 찾아보겠습니다.")
        return head

    parts = [f"I could not stand up any evidence for '{target}'."
             if target else "I could not stand up any evidence."]
    if need_web:
        looked = f"a search for '{query}'" + (f" for {scope}" if scope
                                              else "")
        parts.append(f"{looked} turned up nothing I could verify.")
    else:
        parts.append("It is not in what I know either.")
    parts.append(f"Tell me {axis}"
                 + (f" about {target}" if target else "")
                 + " and I'll search again.")
    return " ".join(parts)


def _realize_clarification(plan: ResponsePlan, lang: str) -> str:
    """The clarification act: WHAT IS MISSING decides what is asked.

    The turn's own ambiguity is the question — the candidates the
    discourse offered when several were in play, the referring
    expression when nothing was there to point it at.  One fixed
    ask-back for every unclear turn would say nothing about which turn
    is unclear, so it is never used."""
    ref = dict(plan.reference or {})
    surface = str(ref.get("surface_form") or "").strip()
    target = (plan.target or "").strip().rstrip("??.!")
    candidates: list[str] = []
    for c in (ref.get("candidates") or []):
        label = str((c or {}).get("target") or "").strip() \
            if isinstance(c, dict) else str(c or "").strip()
        if label and label.lower() != (target or "").lower() \
                and label.lower() not in [x.lower() for x in candidates]:
            candidates.append(label)
    candidates = candidates[:3]
    ko = str(lang or plan.requested_language or "").startswith("ko")
    casual = plan.speech_style == "casual"

    if len(candidates) >= 2:
        joined = ", ".join(candidates[:-1]) + f" or {candidates[-1]}" \
            if len(candidates) >= 3 else " or ".join(candidates)
        if ko:
            return (f"{surface or '그 말'}이 여러 개를 가리킬 수 있어서 "
                    f"확인하고 싶어요 — {joined} 중 어느 쪽인가요?"
                    if casual else
                    f"'{surface or target}'이(가) 여러 후보를 가리켜 "
                    f"확인이 필요합니다 — {joined} 중 어느 쪽이신가요?")
        return (f"I read '{surface or target}' two ways — {joined}. "
                "Which one did you mean?")
    if len(candidates) == 1:
        if ko:
            return (f"{surface or '그 말'}을(를) '{candidates[0]}'로 "
                    f"읽었는데, 맞나요?" if casual else
                    f"'{surface or target}'을(를) '{candidates[0]}'(으)로 "
                    f"읽었습니다. 맞는지요?")
        return (f"I tied '{surface or target}' to {candidates[0]} — "
                "is that what you meant?")
    if surface:
        if ko:
            return (f"'{surface}'이 지금까지 이야기와 연결되지 않았어. "
                    f"무엇을 가리키는 건지 알려줄래?" if casual else
                    f"'{surface}'이(가) 지금까지의 대화와 연결되지 "
                    f"않았습니다. 무엇을 가리키는지 알려주시겠습니까?")
        return (f"'{surface}' does not connect to anything we have "
                "discussed yet — what should I take it to refer to?")
    if target:
        if ko:
            return (f"'{target}'만으로는 무엇을 답할지 정하지 못했어. "
                    "어느 쪽이 궁금한지 알려줄래?" if casual else
                    f"'{target}'만으로는 무엇을 답할지 확정하지 "
                    f"못했습니다. 어느 쪽이 궁금하신지 알려주시겠습니까?")
        return (f"I do not have enough to go on for '{target}' — "
                "which part of it are you after?")
    # Nothing at all to resolve: the missing element IS the subject —
    # the turn names no topic, so the ask-back says which piece is
    # absent rather than asking vaguely for "more".
    _discourse = str((ref.get("discourse_topic") or "").strip())
    if ko:
        head = ("무슨 얘기인지 좀 더 알려주면 정확히 잡을게." if casual
                else "무엇을 이야기하고 있는지 좀 더 알려주시면 "
                     "정확히 이해하겠습니다.")
        if _discourse:
            head = (f"'{_discourse}' 이야기를 이어가는 중인데, "
                    f"그중 무엇을 물어본 건지 알려줄래?"
                    if casual else
                    f"'{_discourse}' 이야기를 이어가는 중입니다. "
                    f"그중 무엇을 물어보신 것인지 알려주시겠습니까?")
        return head
    head = ("That question names no subject to answer about — "
            "tell me what it is on and I will take it from there."
            if not casual else
            "That question names no subject — tell me what it is on "
            "and I'll take it from there.")
    if _discourse:
        head = (f"We are on '{_discourse}' — which part of it are you "
                "after?" if not casual else
                f"We are on '{_discourse}' — which part of it do you "
                "want?")
    return head


def _procedural_head(plan: ResponsePlan) -> str:
    """The procedure's opening line, composed from the target the turn
    actually asked about — never a fixed recipe intro."""
    target = re.sub(
        r"^\s*(the\s+)?(specific\s+)?"
        r"(way|procedure|method|process|steps?|instructions?|guide|"
        r"recipe|order|directions?)\s+(to|for|of|in)\s+",
        "", plan.target or "", flags=re.IGNORECASE).strip(" ??.").strip()
    target = target or (plan.target or "").strip(" ??.").strip()
    ko = str(plan.requested_language or "").startswith("ko")
    casual = plan.speech_style == "casual"
    if ko:
        if not target:
            return "방법은 다음과 같습니다:" if not casual \
                else "방법은 이거야:"
        # the target may already be a verb phrase ("요리하는"): adding
        # another relativizer would double it
        if target.endswith(("하는", "하", "되", "된", "할")):
            return (f"{target} 방법이야:" if casual
                    else f"{target} 방법은 다음과 같습니다:")
        return (f"{target} 하는 방법이야:" if casual
                else f"{target} 하는 방법은 다음과 같습니다:")
    if not target:
        return "Here is the procedure, step by step:"
    verb_like = bool(re.match(
        r"^(to\s+)?(a\s+|an\s+|the\s+)?(make|create|build|write|"
        r"install|configure|set|run|use|cook|prepare|solve|find|"
        r"sort|list|add|remove|delete|fix|check|test|debug|deploy|"
        r"download|open|start|stop|learn|choose|pick|select|do|"
        r"perform|apply|compute|calculate|generate|implement|"
        r"design|plan|organize|manage|maintain|handle|control|"
        r"launch|publish|share|edit|update|upgrade|pair|browse)",
        target, re.IGNORECASE))
    if verb_like:
        return f"Here is how to {target}:"
    return f"Here is the procedure, step by step, for {target}:"


def _append_attribution(parts: list, plan: ResponsePlan) -> None:
    """The evidence's provenance, stated after the claims it supports —
    every retrieved claim carries where it came from."""
    if plan.frame != "web_summary":
        return
    for ev in plan.evidence:
        if ev.startswith(("wikipedia:", "web:", "http")):
            parts.append(f"Source: {ev.split(': ', 1)[-1]}")
            break


def realize_plan(plan: ResponsePlan) -> str:
    """Realize the ResponsePlan into natural language in the requested
    language's register.  The claims are COMPOSED per the intent — a
    definition, a comparison, an assessment — never echoed and never a
    canned answer: the same evidence composes differently per the
    question."""
    reseed()  # generation-time choices re-roll per realization
    lang = plan.requested_language
    claims = [c for c in plan.claims if c]
    intent = plan.intent

    # The conversation acts (greeting / social): no evidence needed —
    # a social reply composed from the semantic structure, never the
    # no-knowledge clarification
    if intent in ("greeting", "salutation", "gratitude", "valediction",
                  "conversation", "state_inquiry") \
            and not claims:
        return _realize_social(plan, intent, lang)

    if intent == "clarification":
        return _realize_clarification(plan, lang)

    if intent == "opinion_request":
        return _realize_opinion(plan, claims, lang)

    if not claims and not plan.answer:
        # No evidence (local AND web): the honest reply states WHAT was
        # sought and WHERE it was sought — the requirement and the
        # retrieval decide the sentence, never a canned "couldn't find
        # it" and never an angle list the turn never asked for.
        return _no_evidence_reply(plan, lang)

    # An investigation's finding: what the sources said, what the
    # search did and where it came from — the same evidence any other
    # knowledge act composes from, realized as the finding it is.
    if intent == "research":
        return _realize_research(plan, claims, lang)

    if lang == "ko":
        return _realize_korean(plan, claims)
    return _realize_english(plan, claims)


def _realize_opinion(plan: ResponsePlan, claims: list, lang: str) -> str:
    """The stance act's reply: the claim is taken seriously, the
    evidence states what it actually supports, and the judgement on top
    of the evidence is left where it belongs.

    With no evidence the reply says so and asks what to look into — it
    never agrees or disagrees blindly, and it never answers the claim
    with a definition of the topic the claim mentions."""
    position = str(plan.proposition or plan.target or "").strip()
    parts: list[str] = []
    if lang == "ko":
        if position:
            parts.append(f"주장을 그대로 두면: {position}")
        if claims:
            parts.append("제가 확인한 근거는 이렇습니다.")
            parts.extend(claims[:3])
            parts.append("근거가 뒷받침하는 범위는 여기까지고, "
                         "그 위의 판단은 고객님이 내리셔야 합니다.")
        else:
            parts.append("근거를 찾지 못해 동의도 반대도 하지 않겠습니다.")
            parts.append("무엇을 확인해 보면 좋을지 알려주시면 "
                         "조사해서 답하겠습니다.")
    else:
        if position:
            parts.append(f"Let's take the claim seriously: {position}")
        if claims:
            parts.append("Here is what the evidence I found actually "
                         "supports.")
            parts.extend(claims[:3])
            parts.append("That is as far as the evidence goes — the "
                         "judgement on top of it is yours.")
        else:
            parts.append("I could not stand behind this with evidence, "
                         "so I will neither agree nor disagree blindly.")
            parts.append("Tell me what to look into and I'll check it.")
    if not claims:
        return "\n".join(p for p in _terminate_parts(parts) if p)
    for ev in (plan.evidence or []):
        if str(ev).startswith(("wikipedia:", "web:", "http")):
            parts.append(f"Source: {str(ev).split(': ', 1)[-1]}")
            break
    return "\n".join(p for p in _terminate_parts(parts) if p)


def _realize_research(plan: ResponsePlan, claims: list,
                      lang: str) -> str:
    """The investigation's finding: the head says what was found out,
    the claims state the evidence as the reasoning composed it, the
    search record says how it was sought and the sources say where it
    came from.  Nothing here is a canned answer — every sentence is the
    evidence's."""
    research = dict(getattr(plan, "research", None) or {})
    subject = str(research.get("subject") or plan.target or "").strip()
    subject = re.sub(r"^(a|an|the)\s+", "", subject,
                     flags=re.IGNORECASE).strip()
    parts: list[str] = []
    if lang == "ko":
        if subject:
            parts.append(f"{subject}에 대해 조사한 결과입니다.")
        parts.extend(claims[:6])
        attempts = int(research.get("attempts") or 1)
        if attempts > 1:
            parts.append(f"첫 검색의 근거가 부족해 쿼리를 다시 구성해 "
                         f"{attempts}회 검색했습니다.")
    else:
        if subject:
            parts.append(f"Here is what I found out about {subject}.")
        parts.extend(claims[:6])
        attempts = int(research.get("attempts") or 1)
        if attempts > 1:
            parts.append(
                "The first results were thin, so I reformulated the "
                f"query and searched again ({attempts} attempts in "
                "total).")
    for ev in (plan.evidence or []):
        if str(ev).startswith(("wikipedia:", "web:", "http")):
            parts.append(f"Source: {str(ev).split(': ', 1)[-1]}")
            break
    return "\n".join(p for p in _terminate_parts(parts) if p)


def _has_batchim(word: str) -> bool:
    """The topic's final sound carries a batchim (Korean) or ends in a
    consonant letter (latin) → the batchim particle (은/이/을);
    otherwise the plain particle (는/가/를)."""
    w = (word or "").strip().rstrip("??.!~")
    if not w:
        return False
    last = w[-1]
    if "\uac00" <= last <= "\ud7a3":  # Hangul syllable
        return (ord(last) - 0xAC00) % 28 != 0
    if last.isalpha():
        return last.lower() not in "aeiou"
    return False


def _terminate_parts(parts: list) -> list:
    """Every part states a full sentence: a claim composed without its
    final punctuation must not run into the part that follows it
    ("…positions in space Space is a …").  Punctuation is the
    sentence's own, never a content decision."""
    out = []
    for i, part in enumerate(parts):
        text = str(part)
        stripped = text.rstrip()
        if i and stripped and stripped[-1] not in ".!?。！？…\")]'”":
            stripped += "."
        out.append(stripped)
    return out


def _realize_english(plan: ResponsePlan, claims: list) -> str:
    parts = []
    intent = plan.intent
    if intent == "capability_query":
        # The capability answer COMPOSED as discourse: the registry's
        # own capabilities in one flowing sentence, never a fixed list.
        if claims:
            parts.append("Here is what I can do — "
                         + "; ".join(claims[:4])
                         + ".")
            parts.append(choice([
                "What are you working on?",
                "Tell me what you want to build and I'll start there.",
            ]))
    elif intent == "language_capability_query":
        if plan.answer:
            head = "Yes." if plan.answer.startswith("yes") else \
                ("Not by default." if plan.answer.startswith("not") else "")
            parts.append(head)
        for c in claims:
            parts.append(c)
    elif intent == "comparison":
        # The contrast composed from both concepts' evidence
        names = []
        for c in claims:
            m = re.match(r"([A-Za-z+#]+):\s+(.+)", c)
            names.append((m.group(1), m.group(2)) if m else ("", c))
        if len(names) == 2:
            parts.append(f"{names[0][0]}: {names[0][1]}")
            parts.append(f"{names[1][0]}: {names[1][1]}")
        else:
            parts.extend(claims)
    elif intent == "suitability_query":
        if plan.answer == "yes":
            parts.append("Yes — based on what I know:")
        elif plan.answer:
            parts.append(plan.answer.capitalize() if plan.answer[0].isalpha()
                         else plan.answer)
        for c in claims[:2]:
            parts.append(f"- {c}")
    elif intent in ("information_query", "explanation_request",
                    "simplify_explanation", "procedural_instruction"):
        # The definitional composition: the first claim becomes "X is
        # ..."; the related evidence follows as context.  The
        # description's form decides the join: a verb-initial
        # description ("Compute the arithmetic mean") is a how-to —
        # "X: <how>" composes correctly where "X is compute..." would
        # break the grammar.
        if intent == "procedural_instruction" and plan.steps:
            # The procedure: the evidence's own steps, in their order,
            # under a head composed from what was asked — never the
            # definition, never an invented sequence.
            parts.append(_procedural_head(plan))
            parts.extend(f"{i}. {s}"
                         for i, s in enumerate(plan.steps, 1))
            _append_attribution(parts, plan)
        elif plan.frame == "web_summary" and claims:
            # The reasoning already composed these claims from the
            # retrieved evidence (framing included): they are stated AS
            # COMPOSED, with their attribution — the source's own text
            # is never emitted.
            parts.extend(claims)
            for ev in plan.evidence:
                if ev.startswith(("wikipedia:", "web:", "http")):
                    parts.append(f"Source: {ev.split(': ', 1)[-1]}")
                    break
        elif plan.target:
            topic = re.sub(r"^(a|an|the)\s+", "", plan.target,
                           flags=re.IGNORECASE).rstrip("?？.")
            if claims:
                body = claims[0]
                body = re.sub(r"^[A-Za-z#+]+:\s+", "", body)
                if plan.style == "simple":
                    # the simple register states the evidence's core
                    body = _core(body)
                verb_initial = bool(re.match(
                    r"^(compute|calculate|find|sort|filter|remove|add)",
                    body.lower()))
                # a body that already leads with the topic (a web
                # extract often restates it) stands alone — "X is the
                # X ..." would repeat the name
                lead = re.sub(r"^the\s+", "", body, flags=re.IGNORECASE)
                if plan.frame == "clause":
                    # The topic is the question's own proposition: the
                    # evidence follows it as its content — defining the
                    # sentence ("X is ...") would be nonsense.
                    head = topic[:1].upper() + topic[1:] if topic else body
                    parts.append(f"{head}. {body}" if head else body)
                elif lead.lower().startswith(topic.lower()):
                    parts.append(lead[0].upper() + lead[1:] if lead
                                 else topic)
                elif verb_initial:
                    parts.append(f"{topic}: {body}")
                else:
                    parts.append(f"{topic} is {body[0].lower() + body[1:]}"
                                 if body else f"{topic}")
            for c in claims[1:]:
                # the same evidence, continued: its sentences — a
                # neighbouring concept is labeled as such instead
                parts.append(c)
            for c in plan.related[:1]:
                parts.append(f"Related: {c}")
            for ev in plan.evidence:
                if ev.startswith(("wikipedia:", "web:")):
                    parts.append(f"Source: {ev.split(': ', 1)[-1]}")
                    break
        else:
            parts.extend(claims)
        if plan.style in ("simple", "brief") and len(parts) > 2:
            parts = parts[:2]
    elif intent in ("recommendation_request", "proposal_request"):
        # The recommendation COMPOSED as guidance: reasoning over the
        # option structure, never a knowledge-dump listing. The claims
        # list is the fallback only when no structured options exist.
        composed = _compose_recommendation_en(plan)
        if composed:
            parts.extend(composed)
        elif claims:
            parts.append("Here's what I can ground that on:")
            for c in claims[:4]:
                parts.append(f"- {c}")
            parts.append("Tell me which direction to take and I'll go deeper.")
    elif intent == "command_query":
        if plan.answer:
            parts.append(f"On {plan.answer}:")
        for c in claims[:3]:
            parts.append(f"- {c}" if ":" in c else c)
        if not claims:
            parts.append("I could not pin down a single command for that "
                         "yet — tell me the platform and the shell and "
                         "I'll narrow it to one.")
    elif intent == "identity":
        parts.append(plan.answer or f"I'm {plan.system_name or 'RENA-BR'}.")
    elif intent == "analysis_request":
        parts.append(plan.answer or "No program was provided to analyze.")
    else:
        # The conversation act: a social reply
        if intent in ("greeting", "salutation", "gratitude",
                      "valediction", "conversation", "state_inquiry"):
            parts.append(_realize_social(plan, intent, "en"))
    return "\n".join(p for p in _terminate_parts(parts) if p)


def _first_clause(text: str, limit: int = 110) -> str:
    """The option's essential purpose: the first semantic clause of its
    description (knowledge text, trimmed for a sentence's flow)."""
    if not text:
        return ""
    for sep in (";", ".", "\n"):
        head = text.split(sep)[0].strip()
        if head:
            text = head
            break
    if len(text) > limit:
        cut = text[:limit]
        # Trim at the last clause boundary — a comma or a word break —
        # so the shortened purpose still reads as a phrase.
        for sep in (", ", " and ", " "):
            idx = cut.rfind(sep)
            if idx > limit // 2:
                cut = cut[:idx]
                break
        text = cut.rstrip(",;")
    return text


def _listing(names: list) -> str:
    """A natural enumeration: 'A, B and C'."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _compose_recommendation_en(plan: ResponsePlan) -> list:
    """The recommendation COMPOSED as guidance — reasoning over the
    option structure (what kind of thing the options are, their
    prerequisite order), never a knowledge-dump listing."""
    options = plan.options or []
    if not options:
        return []
    head = options[0]
    names = [o.get("name", "") for o in options[:3] if o.get("name")]
    cat = head.get("category", "")
    short = _first_clause(head.get("description", ""))
    parts: list = []
    if cat == "programming_language":
        parts.append(f"Of the languages I can ground that on, "
                     f"I'd start with {names[0]}: {short}.")
        if len(options) > 1:
            alt = _first_clause(options[1].get("description", ""))
            parts.append(f"{options[1].get('name')} is a strong alternative "
                         f"if that fits you better — {alt}.")
        parts.append(choice([
            "Tell me what you want to build and I'll narrow it down.",
            "What do you want to make first? That usually settles it.",
        ]))
    elif cat in ("language_fundamental", "language_feature"):
        parts.append(f"Start with the fundamentals — {_listing(names)}. "
                     "They're what everything else builds on.")
        if short:
            parts.append(f"Think of {names[0]} as "
                         f"{short[0].lower() + short[1:]}.")
        parts.append(choice([
            "Once those feel solid, tell me what you want to build "
            "and we'll pick the next step.",
            "Get comfortable with those first, then a small project "
            "makes them stick.",
        ]))
    else:
        parts.append(f"That comes down to {_listing(names)} — "
                     "the pieces the knowledge base connects to this.")
        parts.append(choice([
            "Tell me which one to start from and I'll go deeper.",
            "Pick the closest one and we can start there.",
        ]))
    return parts


def _compose_recommendation_ko(plan: ResponsePlan, casual: bool) -> list:
    options = plan.options or []
    if not options:
        return []
    head = options[0]
    names = [o.get("name", "") for o in options[:3] if o.get("name")]
    cat = head.get("category", "")
    short = _first_clause(head.get("description", ""))
    parts: list = []
    if cat == "programming_language":
        parts.append(f"제가 아는 언어 중에서는 {names[0]}부터 시작하겠어요: {short}."
                     if not casual else
                     f"내가 아는 언어 중엔 {names[0]}부터 추천해: {short}.")
        if len(options) > 1:
            alt = _first_clause(options[1].get("description", ""))
            parts.append(f"{options[1].get('name')}도 좋은 대안이에요 — {alt}."
                         if not casual else
                         f"{options[1].get('name')}도 괜찮아 — {alt}.")
        parts.append("뭘 만들고 싶은지 알려주시면 좁혀드릴게요." if not casual
                     else "뭘 만들고 싶은지 말해주면 좁혀줄게.")
    elif cat in ("language_fundamental", "language_feature"):
        parts.append(f"기초부터 잡는 게 좋습니다 — {_listing(names)}. "
                     "나머지는 전부 이 위에 세워지니까요."
                     if not casual else
                     f"기초부터 하자 — {_listing(names)}. "
                     "나머지는 전부 이 위에 세워지니까.")
        parts.append("기초가 익숙해지면 만들고 싶은 걸 말씀해주세요. "
                     "다음 단계를 같고 골라드릴게요." if not casual else
                     "기초 익숙해지면 만들고 싶은 거 말해줘. 다음 단계 같이 골라줄게.")
    else:
        parts.append(f"여기서는 {_listing(names)}가 핵심이에요." if not casual
                     else f"여기선 {_listing(names)}가 핵심이야.")
        parts.append("어디부터 시작할지 알려주시면 더 깊게 들어가죠." if not casual
                     else "어디부터 시작할지 말해주면 더 파고들게.")
    return parts


def _realize_korean(plan: ResponsePlan, claims: list) -> str:
    parts = []
    intent = plan.intent
    casual = plan.speech_style == "casual"
    if intent == "capability_query":
        if claims:
            parts.append("할 수 있는 일은 이런 것들이야 — " if casual
                         else "지원하는 기능은 이렇습니다 — ")
            parts[-1] += "; ".join(claims[:4]) + "."
            parts.append("지금 뭘 만들고 계신가요?" if not casual
                         else "지금 뭘 만들고 있어?")
    elif intent == "language_capability_query":
        if plan.answer == "네":
            parts.append("응, 가능해." if casual else "네, 가능합니다.")
        elif plan.answer:
            parts.append(plan.answer)
        for c in claims:
            parts.append(f"- {c}")
    elif intent == "comparison":
        for c in claims:
            parts.append(c)
    elif intent == "suitability_query":
        if plan.answer == "네":
            parts.append("응 — 내가 아는 한에서는:" if casual
                         else "네 — 알고 있는 범위에서는:")
        elif plan.answer:
            parts.append(plan.answer)
        for c in claims[:2]:
            parts.append(f"- {c}")
    elif intent in ("information_query", "explanation_request",
                    "simplify_explanation", "procedural_instruction"):
        if intent == "procedural_instruction" and plan.steps:
            # the evidence's own steps, in order, under a head
            # composed from the target — never an invented sequence
            parts.append(_procedural_head(plan))
            parts.extend(f"{i}. {s}"
                         for i, s in enumerate(plan.steps, 1))
            if plan.frame == "web_summary":
                for ev in plan.evidence:
                    if ev.startswith(("wikipedia:", "web:", "http")):
                        parts.append(f"출처: {ev.split(': ', 1)[-1]}")
                        break
        elif plan.frame == "web_summary" and claims:
            parts.extend(claims)
            for ev in plan.evidence:
                if ev.startswith(("wikipedia:", "web:", "http")):
                    parts.append(f"출처: {ev.split(': ', 1)[-1]}")
                    break
        elif plan.target:
            topic = re.sub(r"^(a|an|the)\s+", "", plan.target,
                           flags=re.IGNORECASE).rstrip("?？.")
            if claims:
                body = claims[0]
                body = re.sub(r"^[A-Za-z+#]+:\s+", "", body)
                if plan.style == "simple":
                    body = _core(body)
                if plan.frame == "clause":
                    # the topic is the question's own proposition: the
                    # evidence follows it, it is not defined by it
                    parts.append(f"{topic}. {body}")
                else:
                    particle = "은" if _has_batchim(topic) else "는"
                    parts.append(f"{topic}{particle} {body}")
            for c in claims[1:]:
                parts.append(c)
            for c in plan.related[:1]:
                parts.append(f"관련: {c}")
            for ev in plan.evidence:
                if ev.startswith(("wikipedia:", "web:")):
                    parts.append(f"출처: {ev.split(': ', 1)[-1]}")
                    break
        else:
            parts.extend(claims)
        if plan.style in ("simple", "brief") and len(parts) > 2:
            parts = parts[:2]
    elif intent == "identity":
        if plan.answer:
            parts.append(plan.answer)
        else:
            n = plan.system_name or "RENA-BR"
            parts.append((f"난 {n}이야." if _has_batchim(n) else f"난 {n}야.")
                         if casual else f"저는 {n}입니다.")
    elif intent in ("recommendation_request", "proposal_request"):
        composed = _compose_recommendation_ko(plan, casual)
        if composed:
            parts.extend(composed)
        elif claims:
            parts.append("제가 아는 범위에서 근거가 되는 항목입니다:" if casual
                         else "제가 확인한 근거는 다음과 같습니다:")
            for c in claims[:4]:
                parts.append(f"- {c}")
            parts.append("어느 방향으로 이어갈지 알려주시면 더 파고들겠습니다.")
    elif intent == "command_query":
        if plan.answer:
            parts.append(f"{plan.answer} 환경에서:")
        for c in claims[:3]:
            parts.append(f"- {c}" if ":" in c else c)
        if not claims:
            parts.append("플랫폼과 셸을 알려주시면 하나로 좁혀드리겠습니다."
                         if casual else
                         "플랫폼과 셸을 알려주시면 명령 하나로 좁혀드리겠습니다.")
    elif intent == "analysis_request":
        parts.append(plan.answer or ("분석할 코드가 안 왔어." if casual
                                     else "분석할 코드가 제공되지 않았습니다."))
    else:
        if intent in ("greeting", "salutation", "gratitude",
                      "valediction", "conversation", "state_inquiry"):
            parts.append(_realize_social(plan, intent, "ko"))
    return "\n".join(p for p in _terminate_parts(parts) if p)
