"""Response representation and sentence planning.

The final response is RE-COMPOSED from the turn's understanding —
never copied from a retrieval:

    Retrieved Knowledge → Reasoning → Response Representation
        → Sentence Planning → Language Realization → Response

`ResponseRepresentation` records what the response is to do (its moves
and their sources); `plan_sentences` turns those moves into sentence
units with a function and a budget; the realization then states them
in the requested language.  Evidence enters as CONTENT of a move, so a
source's own text can never become the response by construction.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

# The move roles: what a sentence unit DOES in the response.
ROLE_FRAMING = "framing"          # how the response takes up the request
ROLE_CORE = "core"                # the reasoned content
ROLE_SUPPORT = "support"          # neighbouring evidence / options
ROLE_ATTRIBUTION = "attribution"  # where the evidence came from
ROLE_CLOSING = "closing"          # the follow-up / offer

# Rhetorical connectives per register — chosen by the sentence planner
# from the MOVE's role and the register, never by the sentence's text.
_CONNECTIVES = {
    "en": {
        ROLE_FRAMING: ["Here is the short version:",
                       "In short,", "Simply put,", "The core idea:"],
        ROLE_CORE: ["", "", "", "That is,"],
        ROLE_SUPPORT: ["Beyond that,", "Also,", "As context,"],
        ROLE_CLOSING: ["", "", ""],
    },
    "ko": {
        ROLE_FRAMING: ["짧게 말하면,", "핵심은,", "쉽게 말해,"],
        ROLE_CORE: ["", "", ""],
        ROLE_SUPPORT: ["추가로,", "참고로,"],
        ROLE_CLOSING: ["", "", ""],
    },
}


@dataclass
class Move:
    """One move of the response: what it says and where it came from."""
    role: str = ROLE_CORE
    content: str = ""
    source: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content,
                "source": self.source,
                "confidence": round(self.confidence, 3)}


@dataclass
class SentencePlan:
    """A planned sentence: its function and its budget."""
    index: int = 0
    function: str = ROLE_CORE
    content: str = ""
    max_length: int = 200

    def to_dict(self) -> dict:
        return {"index": self.index, "function": self.function,
                "content": self.content, "max_length": self.max_length}


@dataclass
class ResponseRepresentation:
    """The response's meaning-level plan (before wording).

    It is composed from everything the turn is: the USER'S GOAL, the
    SEMANTIC REPRESENTATION, the CONVERSATION CONTEXT, the RELEVANT
    KNOWLEDGE, the EVIDENCE, the RESPONSE REQUIREMENTS and the
    UNCERTAINTY — so the generation can check what it owes before it
    summarizes."""
    intent: str = ""
    target: str = ""
    style: str = ""
    language: str = "en"
    reference: dict = field(default_factory=dict)
    topic_state: dict = field(default_factory=dict)
    knowledge: dict = field(default_factory=dict)
    # What the turn is ABOUT at each level: the user's own goal, and
    # the semantic representation the goal was read from.
    goal: dict = field(default_factory=dict)
    semantic: dict = field(default_factory=dict)
    # What the reply MUST carry (the register's bounds, the document's
    # own response requirements, the evidence obligation) — recorded
    # where the generation reads them, never where a retriever could.
    requirements: list = field(default_factory=list)
    # What could NOT be established for this reply: an unresolved
    # reference, an intent whose evidence is missing, a subject the
    # turn never named.  The generation states the gap instead of
    # presenting a summary as if the gap were not there.
    uncertainty: list = field(default_factory=list)
    moves: list[Move] = field(default_factory=list)
    sentence_plan: list[SentencePlan] = field(default_factory=list)
    # The long-form layer's sections and the length budget they obey
    # (empty for the short path — the same representation serves both).
    sections: list[dict] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    confidence: float = 0.0
    needs_clarification: bool = False

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "target": self.target,
            "style": self.style,
            "language": self.language,
            "reference": dict(self.reference),
            "topic_state": dict(self.topic_state),
            "knowledge": dict(self.knowledge),
            "goal": dict(self.goal),
            "semantic": dict(self.semantic),
            "requirements": [dict(r) for r in self.requirements],
            "uncertainty": list(self.uncertainty),
            "moves": [m.to_dict() for m in self.moves],
            "sentence_plan": [s.to_dict() for s in self.sentence_plan],
            "sections": [dict(s) for s in self.sections],
            "budget": dict(self.budget),
            "confidence": round(self.confidence, 3),
            "needs_clarification": self.needs_clarification,
        }


# ---------------------------------------------------------------------
# evidence → claims (REASONING, not copying)
# ---------------------------------------------------------------------

# A trailing ELABORATION: the simple register states the claim's core
# and leaves the qualifying detail behind (a register decision over the
# sentence's structure, never a change of claim).
_TRAILING_ELABORATION = re.compile(
    r"\s+(?:containing|including|covering|spanning|consisting\s+of|"
    r"made\s+of|located\s+in|found\s+in|used\s+for|used\s+in|"
    r"known\s+for|such\s+as|ranging\s+from|extending\s+over)\b.*$",
    re.IGNORECASE)


def _condense(text: str, style: str = "simple") -> str:
    """The evidence sentence's core: parentheticals, appositions and
    trailing elaboration removed for a simple register.  The claim that
    comes back is DERIVED from the sentence — never the sentence itself
    when a simpler core exists."""
    import re as _re
    out = (text or "").strip()
    if not out:
        return out
    out = _re.sub(r"\s*\([^)]*\)", "", out)             # parentheticals
    out = _re.split(r"\s*[;:]\s+", out, maxsplit=1)[0]  # after an apposition
    out = _re.split(r",\s+(?:which|where|who|while|although|because)\b",
                    out, maxsplit=1, flags=_re.IGNORECASE)[0]
    if style in ("simple", "brief"):
        out = _TRAILING_ELABORATION.sub("", out)
    out = out.rstrip(",—- ").strip()
    if not out:
        return (text or "").strip()
    # A sentence that runs on says more than the question asked: for a
    # simple register the claim is the CORE — cut at its own clause
    # boundaries, never carried whole.
    if style in ("simple", "brief") and len(out.split()) > 22:
        acc = ""
        for part in _re.split(r",\s+|;\s+", out):
            candidate = f"{acc}, {part}" if acc else part
            if len(candidate.split()) > 22:
                break
            acc = candidate
        if len(acc.split()) >= 8:
            out = acc.rstrip(" ,;")
    if not out:
        return (text or "").strip()
    if out[-1] not in ".!?":
        out += "."
    return out


def _apply_connective(connective: str, claim: str) -> str:
    """The framing connective joins the claim as ONE sentence: when the
    connective is a comma phrase, a claim that opens with its own
    article continues that sentence — "Simply put, The version …"
    reads as two spliced sentences, not one."""
    joined = f"{connective} {claim}"
    if connective.endswith(",") and claim[:1].isupper():
        head = claim.split(" ", 1)[0]
        if head in {"The", "A", "An", "This", "These", "Those", "It",
                    "They", "There", "That"}:
            joined = f"{connective} {claim[:1].lower()}{claim[1:]}"
    return joined


def compose_evidence_claims(knowledge, target: str, style: str = "",
                            limit: int | None = None) -> list[str]:
    """Reason the ranked evidence into claims: the core of the most
    relevant content, within the register's budget.

    A claim is DERIVED from the evidence.  When the evidence's sentence
    has no simpler core to take (nothing to condense), the claim is
    framed with the register's connective instead — so a claim is never
    the source's own sentence, whatever the source says."""
    if knowledge is None:
        return []
    budget = limit if limit is not None else (1 if style in ("simple",
                                                             "brief") else 2)
    language = getattr(knowledge, "language", "en")
    claims: list[str] = []
    for evidence in knowledge.ranked():
        sentences = list(evidence.relevant_claims) or []
        if not sentences and evidence.content:
            sentences = [evidence.content]
        for sentence in sentences:
            claim = _condense(sentence, style)
            unchanged = claim.strip() == (sentence or "").strip()
            # Either way the claim is the RESPONSE'S OWN statement of
            # the evidence: a simple register announces it (so the
            # simpler form never reads as a repeat of what was said),
            # and a claim nothing could be condensed from is framed so
            # the source's sentence is never echoed.
            if unchanged or style == "simple":
                connective = choose_connective(ROLE_FRAMING, language)
                if connective:
                    claim = _apply_connective(connective, claim)
            if claim and claim not in claims:
                claims.append(_terminate_sentence(claim))
            if len(claims) >= budget:
                return claims
        if claims:
            return claims
    return claims


def _terminate_sentence(text: str) -> str:
    """The claim ends as a sentence: a claim cut from the evidence
    without its final punctuation would otherwise run into whatever
    follows it in the response."""
    stripped = (text or "").rstrip()
    if stripped and stripped[-1] not in ".!?。！？…\")]'”":
        stripped += "."
    return stripped


def frame_claims(target: str, claims: list[str], frame: str = "") -> list[str]:
    """Compose the claims with the topic they answer — the reasoning's
    frame (definition / clause), never a paste of the source."""
    if not claims or not target:
        return claims
    topic = target.strip().rstrip("??.!")
    topic = _strip_article(topic)
    out = []
    for claim in claims:
        body = claim.strip()
        if not body:
            continue
        if frame == "web_summary":
            # the reasoning already composed this claim from the
            # evidence (framing included) — it is stated as it is
            out.append(body)
            continue
        lead = _strip_article(body)
        if lead.lower().startswith(topic.lower()) or frame == "clause":
            out.append(body if frame != "clause"
                       else f"{_capitalize(topic)}. {body}"
                       if not lead.lower().startswith(topic.lower())
                       else body)
        else:
            out.append(f"{_capitalize(topic)} is "
                       f"{body[0].lower() + body[1:]}"
                       if body[0].isalpha() else body)
    return out or claims


def _strip_article(text: str) -> str:
    import re
    return re.sub(r"^(a|an|the)\s+", "", (text or "").strip(),
                  flags=re.IGNORECASE)


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


# ---------------------------------------------------------------------
# representation + sentence planning
# ---------------------------------------------------------------------

# The acts whose reply must STAND ON EVIDENCE: an interaction of these
# kinds that carries none has not answered, so the representation
# records the gap and the generation states it instead of summarizing.
_EVIDENCE_INTENTS = frozenset({
    "information_query", "explanation_request", "simplify_explanation",
    "procedural_instruction", "suitability_query", "analysis_request",
    "comparison", "research", "recommendation_request",
    "proposal_request", "opinion_request", "clarification",
})


def build_representation(plan, semantic=None, reference=None,
                         topic_state=None, knowledge=None) -> ResponseRepresentation:
    """The response's meaning-level plan, from the reasoned content.

    Everything the turn is arrives here: the user's goal and the
    semantic representation it was read from, the conversation's
    context, the relevant knowledge and its evidence, the response
    requirements the request itself stated, and what could not be
    established.  The generation reads the SAME structure, so a gap is
    visible before the reply is worded."""
    rep = ResponseRepresentation(
        intent=str(getattr(plan, "intent", "") or ""),
        target=str(getattr(plan, "target", "") or ""),
        style=str(getattr(plan, "style", "") or ""),
        language=str(getattr(plan, "requested_language", "en") or "en"),
        confidence=float(getattr(plan, "confidence", 0.0) or 0.0),
        reference=reference.to_dict() if hasattr(reference, "to_dict")
        else (dict(reference) if isinstance(reference, dict) else {}),
        topic_state=topic_state.to_dict() if hasattr(topic_state, "to_dict")
        else (dict(topic_state) if isinstance(topic_state, dict) else {}),
        knowledge=knowledge.to_dict() if hasattr(knowledge, "to_dict")
        else (dict(knowledge) if isinstance(knowledge, dict) else {}),
    )
    if reference is not None and getattr(reference, "needs_clarification",
                                         False):
        rep.needs_clarification = True

    _record_goal_and_semantics(rep, plan, semantic)
    _record_requirements(rep, plan, semantic)
    _record_uncertainty(rep, plan)

    # The framing move: the request's own act (its intent and target)
    if rep.target:
        rep.moves.append(Move(role=ROLE_FRAMING, content=rep.target,
                              source="act", confidence=rep.confidence))
    # The core moves: the reasoned claims and where each came from
    for claim in (getattr(plan, "claims", None) or []):
        rep.moves.append(Move(role=ROLE_CORE, content=claim,
                              source="evidence", confidence=rep.confidence))
    # The support moves: neighbouring evidence and the option structure
    for related in (getattr(plan, "related", None) or []):
        rep.moves.append(Move(role=ROLE_SUPPORT, content=related,
                              source="related evidence"))
    for option in (getattr(plan, "options", None) or [])[:4]:
        name = str(option.get("name") or option.get("title") or "")
        if name:
            rep.moves.append(Move(role=ROLE_SUPPORT, content=name,
                                  source="options"))
    # The attribution move: the evidence's provenance (web evidence is
    # always attributed, never silently absorbed)
    for evidence in (getattr(plan, "evidence", None) or []):
        if str(evidence).startswith(("wikipedia:", "web:", "http")):
            rep.moves.append(Move(role=ROLE_ATTRIBUTION,
                                  content=str(evidence),
                                  source=str(evidence)))
            break
    # The uncertainty move: what this reply could not stand behind —
    # carried as a move so the sentence plan sees it, never as a claim
    # (a claim is what the evidence supports; this is what it did not).
    for gap in rep.uncertainty:
        rep.moves.append(Move(role=ROLE_CLOSING, content=gap,
                              source="uncertainty",
                              confidence=rep.confidence))
    return rep


def _record_goal_and_semantics(rep: ResponseRepresentation, plan,
                               semantic) -> None:
    """The user's goal and the semantic representation the reply is
    answering from — recorded, so the representation names what the
    turn wanted, not only what it was categorized as."""
    if semantic is None:
        return
    goal = getattr(semantic, "goal", None)
    goal_type = getattr(goal, "goal_type", None)
    act = getattr(semantic, "dialogue_intent", None) or {}
    rep.goal = {
        "type": str(getattr(goal_type, "value", "") or ""),
        "target": str(getattr(goal, "target", "") or ""),
        "act": str(act.get("goal") or ""),
        "confidence": round(float(getattr(goal, "confidence", 0.0) or 0.0),
                            3),
    }
    rep.semantic = {
        "intent": str(act.get("type") or ""),
        "route": str(act.get("route") or ""),
        "topics": [str(t) for t in (getattr(semantic, "intent_topics", None)
                                    or [])][:8],
        "entities": [str(getattr(e, "name", "")) for e in
                     (getattr(semantic, "entities", None) or [])][:8],
        "language": str(getattr(getattr(semantic, "source_language", None),
                                "value", "") or ""),
    }


def _record_requirements(rep: ResponseRepresentation, plan,
                         semantic) -> None:
    """What the reply MUST carry, from the request's own statement of
    it — the register's bounds, the document's response requirements
    and the evidence obligation of the act.  Recorded where the
    generation reads them, never where a retriever could reach."""
    requirements: list[dict] = []
    spec = getattr(semantic, "response_spec", None) if semantic else None
    spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else (
        dict(spec) if isinstance(spec, dict) else {})
    for constraint in (spec_dict.get("constraints") or []):
        value = str(constraint or "").strip()
        if value:
            requirements.append({"kind": "register", "value": value})
    for key in ("target_length", "minimum_length", "maximum_length"):
        value = spec_dict.get(key)
        if value:
            requirements.append({"kind": "length", "value": str(value)})
    document = getattr(plan, "document", None)
    for requirement in (getattr(document, "response_requirements", None)
                        or []):
        kind = str(getattr(requirement, "kind", "") or "")
        value = str(getattr(requirement, "value", "") or "")
        if kind or value:
            requirements.append({"kind": kind or "content", "value": value})
    if rep.intent in _EVIDENCE_INTENTS:
        requirements.append(
            {"kind": "evidence",
             "value": "the reply states only what the evidence supports"})
    rep.requirements = requirements


def _record_uncertainty(rep: ResponseRepresentation, plan) -> None:
    """What could NOT be established for this reply.

    The gap is recorded structurally — an unresolved reference, an act
    that needs evidence and has none, a subject the turn never named —
    so the generation must state it rather than present a summary as if
    the missing part were there."""
    gaps: list[str] = []
    if rep.needs_clarification:
        gaps.append("the reference did not resolve: the turn asks what "
                    "it points at")
    if rep.intent in _EVIDENCE_INTENTS:
        if not (getattr(plan, "claims", None) or []):
            gaps.append("no evidence: this act needs grounding the "
                        "knowledge did not provide")
        if not rep.target.strip() and rep.intent != "clarification":
            gaps.append("no subject: the turn's topic was not resolved")
    if float(rep.confidence or 0.0) < 0.35 and rep.intent != "clarification":
        gaps.append(f"low confidence: {rep.confidence:.2f}")
    rep.uncertainty = gaps


def plan_sentences(rep: ResponseRepresentation,
                   plan=None) -> list[SentencePlan]:
    """Sentence planning: the moves become sentence units with a
    function and a length budget (the register's)."""
    simple = rep.style in ("simple", "brief")
    budget = 90 if simple else 160
    if simple:
        # the simple register's unit is the claim's CORE (its statement
        # before the elaboration) — the plan and the realization carry
        # the SAME content, so what was planned is what is said.
        from brain.conversation_knowledge import _core
    sentences: list[SentencePlan] = []
    index = 0
    for move in rep.moves:
        if not move.content:
            continue
        content = move.content
        if simple and move.role == ROLE_CORE and rep.style == "simple":
            content = _core(content)
        max_length = budget if move.role in (ROLE_FRAMING, ROLE_CORE) \
            else 120
        sentences.append(SentencePlan(index=index, function=move.role,
                                      content=content,
                                      max_length=max_length))
        index += 1
        if simple and move.role == ROLE_CORE and index >= 3:
            break
    # the plan's own answer (identity / yes-no) is a sentence too
    answer = str(getattr(plan, "answer", "") or "") if plan else ""
    if answer and not any(s.content == answer for s in sentences):
        sentences.append(SentencePlan(index=index, function=ROLE_CORE,
                                      content=answer, max_length=budget))
    rep.sentence_plan = sentences
    return sentences


def realize(rep: ResponseRepresentation, plan) -> str:
    """Language realization: the plan's content stated in the
    requested language's register.

    The realization reads the sentence plan (so what is said is what
    was planned), composing it through the response's intent — the
    retrieved material itself is never the response.  Before the reply
    goes out it is CHECKED against what the representation said the
    reply owes: a requirement the content cannot satisfy is stated as
    the gap it is, never left out of a summary that reads as if it had
    been covered."""
    from brain.conversation_knowledge import realize_plan
    if rep.sentence_plan:
        planned = [s.content for s in rep.sentence_plan
                   if s.function == ROLE_CORE and s.content]
        if planned:
            plan.claims = planned
    if rep.needs_clarification:
        plan.intent = "clarification"
    body = realize_plan(plan)
    return enforce_requirements(rep, plan, body)


def _has_structure(text: str) -> bool:
    """The reply is ORGANIZED: more than one line, a heading or an
    ordered run — the shape "clear sections" asks for."""
    lines = [ln for ln in str(text or "").splitlines() if ln.strip()]
    if len(lines) >= 3:
        return True
    if re.search(r"^\s*(?:\d+[.)]|[-*•])\s+\S", str(text or ""),
                 re.MULTILINE):
        return True
    return bool(re.search(r"^[^\n]{2,60}:\s*$", str(text or ""),
                          re.MULTILINE))


def _covers_quantity(text: str, value: str) -> bool:
    """The reply carries the QUANTITY the request stated ("30 minutes",
    "one month"): the number and its unit both appear — a reply that
    silently drops the quantity did not meet the requirement."""
    quantities = re.findall(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"twenty|thirty|forty|fifty|hundred)\s*[-\s]?"
        r"(?:minute|second|hour|day|week|month|year|word|sentence|line|"
        r"paragraph|page|character|step)s?\b",
        str(value or ""), re.IGNORECASE)
    if not quantities:
        return True
    body = str(text or "").lower()
    for amount in quantities:
        word = str(amount).lower()
        spelled = {"1": "one", "2": "two", "3": "three", "4": "four",
                   "5": "five", "6": "six", "7": "seven", "8": "eight",
                   "9": "nine", "10": "ten", "20": "twenty",
                   "30": "thirty", "40": "forty", "50": "fifty",
                   "100": "hundred"}.get(word, word)
        unit = re.search(
            r"\b" + re.escape(word) + r"\s*[-\s]?"
            r"(minute|second|hour|day|week|month|year|word|sentence|"
            r"line|paragraph|page|character|step)", str(value or ""),
            re.IGNORECASE)
        unit_word = (unit.group(1) + "s") if unit else ""
        if word not in body and spelled not in body:
            return False
        if unit_word and unit_word.rstrip("s") not in body:
            return False
    return True


def unmet_requirements(rep: ResponseRepresentation, plan,
                       body: str) -> list[dict]:
    """The requirements the planned content could NOT satisfy.

    Only the requirements whose satisfaction is a property of the
    WRITTEN reply are judged here — its organization, its examples and
    the quantities it carries.  Everything the representation recorded
    is listed for the trace; these are the ones a summary would hide."""
    out: list[dict] = []
    text = str(body or "")
    for requirement in rep.requirements:
        kind = str(requirement.get("kind") or "")
        value = str(requirement.get("value") or "")
        if not value:
            continue
        if kind == "organization":
            if not _has_structure(text):
                out.append(requirement)
        elif kind == "examples":
            if not re.search(r"\b(?:e\.g\.|example|instance|for "
                             r"instance|예시|예제|사례)\b",
                             text, re.IGNORECASE):
                out.append(requirement)
        elif kind == "length":
            if not _covers_quantity(text, value):
                out.append(requirement)
    return out


def enforce_requirements(rep: ResponseRepresentation, plan,
                         body: str) -> str:
    """The generation's gate: what the representation recorded as owed
    and the content did not deliver is SAID — so a summary never reads
    as if the missing part were covered.

    With nothing owed the reply is returned unchanged; with evidence
    missing the reply already states it (the plan's own no-evidence
    path), so only a requirement that a confident-looking summary would
    otherwise hide is added, and only briefly."""
    if rep.needs_clarification or rep.intent == "clarification":
        return body
    has_claims = bool(getattr(plan, "claims", None) or [])
    language = str(getattr(plan, "requested_language", "en") or "en")
    notes: list[str] = []
    missing = unmet_requirements(rep, plan, body)
    if missing and has_claims:
        names = ", ".join(sorted({str(m.get("value") or m.get("kind") or "")
                                  for m in missing[:2]}))
        notes.append(
            f"Note: the reply does not carry what you asked for here "
            f"({names})."
            if language != "ko"
            else f"참고로, 요청하신 {names}은(는) 이 답에 담지 못했습니다.")
    if not notes:
        return body
    return "\n".join([body] + notes) if body else "\n".join(notes)


def choose_connective(role: str, language: str = "en") -> str:
    """The sentence planner's connective for a move's role and the
    register — a rhetorical choice, not a sentence lookup."""
    options = _CONNECTIVES.get(language or "en", _CONNECTIVES["en"]).get(
        role, [""])
    return random.choice([o for o in options] or [""])
