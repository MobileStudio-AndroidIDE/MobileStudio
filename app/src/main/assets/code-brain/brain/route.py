"""The turn's route: which pipeline answers this utterance.

The route is the structural answer to "does this turn enter the code
pipeline?".  It is decided from what the understanding ALREADY
produced — the parser's goal and the dialogue act's derived intent —
and from nothing else: never a topic word, never a stored answer, never
a sentence→answer map.

    NL → Semantic Understanding → Intent (the ACT)
       → Context / Reference resolution → Topic State → ROUTE

    CODE              artifact request   hypotheses → reasoning → plan → IR → code
    EXPLANATION       explanation/answer knowledge    → reasoning → response
    ADVICE_LEARNING   guidance/options   knowledge    → reasoning → response
    CHAT              everyday talk      conversation context    → response

A non-CODE route never enters the Code Hypothesis Generator: the
understanding of an everyday utterance is not a problem state.
"""

from __future__ import annotations

from enum import Enum


class Route(str, Enum):
    """The pipeline a turn is routed to."""

    CODE = "CODE"
    EXPLANATION = "EXPLANATION"
    ADVICE_LEARNING = "ADVICE_LEARNING"
    CHAT = "CHAT"


# The goals whose pipeline produces an artifact (the CODE route).
_CODE_GOALS = frozenset({
    "create_function",
    "create_class",
    "create_project",
    "modify",
    "fix",
    "computer",
})

# The goals the ACT may override: a learning/explanation/social/
# procedural predication over an artifact goal is DIALOGUE, not
# creation — the request asks to be TOLD, not to have it made.
_ACT_OVERRIDE_GOALS = frozenset({
    "create_function",
    "create_class",
    "create_project",
    "modify",
    "fix",
    "computer",
})

# The derived intent types that carry such a dialogue act (the act, not
# the topic, decides this).
_DIALOGUE_INTENTS = frozenset({
    "conversation",
    "greeting",
    "explanation_request",
    "simplify_explanation",
    "recommendation_request",
    "procedural_instruction",
    "capability_query",
    "language_capability_query",
    "clarification",
    # the two acts that never produce an artifact either: a stance on
    # the turn's own claim and a candidate the turn asked to be given
    "opinion_request",
    "proposal_request",
})

# The dialogue acts answered from knowledge (reasoning → response).
_EXPLANATION_INTENTS = frozenset({
    "explanation_request",
    "simplify_explanation",
    "information_query",
    "command_query",
    "analysis",
    # a procedure request is answered from knowledge too: the ordered
    # way of doing something is reasoned, never generated as code
    "procedural_instruction",
    # an investigation is answered from EVIDENCE, iterated until it is
    # sufficient — the knowledge is not in the conversation yet
    "research",
    # a stance on the turn's own claim is answered from EVIDENCE too:
    # the claim is discussed, never defined away
    "opinion_request",
})

# The dialogue acts answered with guidance (knowledge → reasoning →
# response): learning recommendations, advice and a proposal the
# addressee asked to be given.
_ADVICE_INTENTS = frozenset({"recommendation_request",
                             "proposal_request"})


def route_of_intent(intent: str) -> Route:
    """The pipeline for a turn whose act is not an artifact request."""
    if intent in _ADVICE_INTENTS:
        return Route.ADVICE_LEARNING
    if intent in _EXPLANATION_INTENTS:
        return Route.EXPLANATION
    return Route.CHAT


def decide_route_parts(goal_value: str, intent: str) -> Route:
    """The route from the two values the understanding already carries:
    the parser's goal type and the act's derived intent type."""
    if goal_value in _CODE_GOALS:
        if goal_value in _ACT_OVERRIDE_GOALS and intent in _DIALOGUE_INTENTS:
            return route_of_intent(intent)
        return Route.CODE
    return route_of_intent(intent)


def decide_route(semantic) -> Route:
    """The turn's route — a pure function of the understanding.

    `semantic` is the SemanticRepresentation (or anything exposing a
    `goal.goal_type` and a `dialogue_intent` dict).  The intent type is
    read from the already-derived dialogue intent; an artifact goal
    stays CODE unless the ACT overrode it.
    """
    dialogue_intent = getattr(semantic, "dialogue_intent", None) or {}
    stamped = str(dialogue_intent.get("route") or "")
    if stamped:
        # The route was already decided for this turn (at the act
        # layer, or by the structural problem gate): every consumer
        # reads the SAME decision.
        try:
            return Route(stamped)
        except ValueError:
            pass
    goal = getattr(getattr(semantic, "goal", None), "goal_type", None)
    goal_value = getattr(goal, "value", "") or ""
    intent = str(dialogue_intent.get("type") or "")
    return decide_route_parts(goal_value, intent)


def describe_route(route: "Route | str", intent: str = "") -> str:
    """The route's pipeline, for the trace.

    With the turn's INTENT the label names the pipeline the request
    actually entered — a procedure request, an example request, a
    comparison — while the route stays the structural decision it
    always was (does this turn produce an artifact?).
    """
    value = getattr(route, "value", route)
    pipelines = {
        "procedural_instruction":
            "semantic → intent → context → knowledge → evidence → "
            "procedure → response",
        "information_query":
            "semantic → intent → context → knowledge → evidence → "
            "response",
        "explanation_request":
            "semantic → intent → context → knowledge → evidence → "
            "response",
        "simplify_explanation":
            "semantic → intent → context → knowledge → evidence → "
            "simpler response",
        "recommendation_request":
            "semantic → intent → context → knowledge → options → "
            "response",
        "comparison":
            "semantic → intent → context → knowledge → contrast → "
            "response",
        "research":
            "semantic → intent → research loop → evidence → "
            "iteration → synthesis → response",
        "capability_query":
            "semantic → intent → self knowledge → response",
        "clarification":
            "semantic → intent → context → missing structure → "
            "clarification",
        "conversation":
            "semantic → intent → conversation context → response",
        "opinion_request":
            "semantic → intent → context → claim → evidence → "
            "stance → response",
        "proposal_request":
            "semantic → intent → context → knowledge → options → "
            "proposal → response",
    }
    if intent in pipelines and value != Route.CODE.value:
        return pipelines[intent]
    return {
        Route.CODE.value: "hypotheses → reasoning → plan → IR → code",
        Route.EXPLANATION.value: "knowledge → reasoning → response",
        Route.ADVICE_LEARNING.value: "knowledge → reasoning → response",
        Route.CHAT.value: "conversation context → response",
    }.get(value, "conversation context → response")
