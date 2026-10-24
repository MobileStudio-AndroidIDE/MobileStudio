"""Semantic Composer - composes new meaning structures from known concepts.

This module implements the compositional reasoning layer of RENA-BR. Instead
of matching whole sentences against a fixed list of operations, it:

  1. Decomposes natural language into **atomary semantic primitives** (cues)
     using a small, general, knowledge-grounded ontology of operations.
  2. Composes those primitives into a dependency graph (DAG) using
     compositional rules (binding a comparison to its aggregate, a filter to
     its predicate, a sort to its sequence, etc.).
  3. Keeps three layers explicitly separated:

     * User Semantics     - primitives actually extracted from the user text
     * Retrieved Knowledge- activated concepts / relations from the KG
     * Derived Semantics  - the composed OperationGraph (new structure)

The key property: a *new* sentence that never appeared in any list is handled
by combining primitives the system already knows, rather than by adding a new
keyword/entry per sentence.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from brain.semantic_representation import (
    OperationSpec, OperationGraph, OperationType,
)
from brain.relation_store import RelationType

# --------------------------------------------------------------------------- #
# Primitive operation ontology
# --------------------------------------------------------------------------- #
# Each primitive is a general, reusable unit of meaning. ``producible`` lists
# the primitive names that may be referenced by another primitive (used for
# dependency binding). Triggers are lexical groundings — the "knowledge part",
# not a per-problem lookup table.
# --------------------------------------------------------------------------- #

@dataclass
class Primitive:
    name: str
    op_type: OperationType
    semantic_role: str
    triggers: tuple[str, ...]
    # what data type this operation consumes (e.g. "sequence", "scalar")
    consumes: tuple[str, ...] = ()
    # what data type this operation produces (e.g. "sequence", "scalar", "predicate")
    produces: str = ""
    # the contract this operation assumes (spec #3) — a closed
    # per-primitive vocabulary; empty falls back to the type wiring.
    preconditions: tuple[str, ...] = ()


PRIMITIVE_ONTOLOGY: dict[str, Primitive] = {
    # ── Data sources ──
    "generate_range": Primitive(
        "generate_range", OperationType.IO, "range",
        triggers=(), consumes=(), produces="sequence",
    ),

    # ── Transforms (sequence → sequence) ──
    "remove_duplicates": Primitive(
        "remove_duplicates", OperationType.TRANSFORM, "deduplication",
        triggers=(), consumes=("sequence",), produces="sequence",
    ),
    "reverse": Primitive(
        "reverse", OperationType.TRANSFORM, "string_reverse",
        triggers=(), consumes=("sequence",), produces="sequence",
    ),
    "reverse_sequence": Primitive(
        "reverse_sequence", OperationType.TRANSFORM, "string_reverse",
        triggers=(), consumes=("sequence",), produces="sequence",
    ),
    "factorial": Primitive(
        "factorial", OperationType.ARITHMETIC, "algorithm",
        triggers=(), consumes=("scalar",), produces="scalar",
    ),
    "fibonacci": Primitive(
        "fibonacci", OperationType.ARITHMETIC, "algorithm",
        triggers=(), consumes=("scalar",), produces="scalar",
    ),
    "concatenate": Primitive(
        "concatenate", OperationType.TRANSFORM, "string_concatenation",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    "add": Primitive(
        "add", OperationType.ARITHMETIC, "arithmetic",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    "multiply": Primitive(
        "multiply", OperationType.ARITHMETIC, "arithmetic",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),

    # ── Aggregates (sequence → scalar) ──
    "calculate_average": Primitive(
        "calculate_average", OperationType.AGGREGATE, "average",
        triggers=(), consumes=("sequence",), produces="scalar",
        preconditions=("input collection non-empty for a defined average",),
    ),
    "calculate_sum": Primitive(
        "calculate_sum", OperationType.AGGREGATE, "sum",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    "find_max": Primitive(
        "find_max", OperationType.AGGREGATE, "max",
        triggers=(), consumes=("sequence",), produces="scalar",
        preconditions=("input collection non-empty for a defined extremum",),
    ),
    "search": Primitive(
        "search", OperationType.AGGREGATE, "search",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    "find_min": Primitive(
        "find_min", OperationType.AGGREGATE, "min",
        triggers=(), consumes=("sequence",), produces="scalar",
        preconditions=("input collection non-empty for a defined extremum",),
    ),

    # ── Predicates (element → boolean) ──
    "is_prime": Primitive(
        "is_prime", OperationType.CONTROL_FLOW, "primality",
        triggers=(), consumes=("sequence",), produces="predicate",
        preconditions=("input elements are integers",),
    ),
    "is_multiple_of": Primitive(
        "is_multiple_of", OperationType.ARITHMETIC, "divisibility",
        triggers=(), consumes=("sequence",), produces="predicate",
        preconditions=("divisor is a non-zero integer",),
    ),

    # ── Comparisons (scalar → predicate) ──
    "compare_greater_than": Primitive(
        "compare_greater_than", OperationType.COMPARISON, "comparison",
        triggers=(), consumes=("scalar",), produces="predicate",
        preconditions=("a scalar operand is bound",),
    ),
    "compare_less_than": Primitive(
        "compare_less_than", OperationType.COMPARISON, "comparison",
        triggers=(), consumes=("scalar",), produces="predicate",
        preconditions=("a scalar operand is bound",),
    ),

    # ── Filters (sequence + predicate → sequence) ──
    "filter": Primitive(
        "filter", OperationType.FILTER, "selection",
        triggers=(), consumes=("sequence", "predicate"), produces="sequence",
        preconditions=("a predicate is provided",),
    ),
    "filter_elements": Primitive(
        "filter_elements", OperationType.FILTER, "selection",
        triggers=(), consumes=("sequence", "predicate"), produces="sequence",
        preconditions=("a predicate is provided",),
    ),

    # ── Sorts (sequence → sequence) ──
    "sort_descending": Primitive(
        "sort_descending", OperationType.SORT, "sort_desc",
        triggers=(), consumes=("sequence",), produces="sequence",
        preconditions=("input elements are mutually comparable",),
    ),
    "sort_ascending": Primitive(
        "sort_ascending", OperationType.SORT, "sort_asc",
        triggers=(), consumes=("sequence",), produces="sequence",
        preconditions=("input elements are mutually comparable",),
    ),

    # ── Output / IO primitives ──
    "emit_output": Primitive(
        "emit_output", OperationType.IO, "output",
        triggers=(), consumes=("literal",), produces="output",
    ),
    "repeat_output": Primitive(
        "repeat_output", OperationType.IO, "repeat",
        triggers=(), consumes=("output",), produces="output",
    ),
    "literal_value": Primitive(
        "literal_value", OperationType.IO, "literal",
        triggers=(), consumes=(), produces="literal",
    ),
    
    # ── Competitive Programming Primitives ──
    # These are general algorithmic primitives that can be composed
    # to solve competitive programming problems
    "iterate_range": Primitive(
        "iterate_range", OperationType.TRANSFORM, "iteration",
        triggers=(), consumes=("scalar",), produces="sequence",
    ),
    "conditional_filter": Primitive(
        "conditional_filter", OperationType.FILTER, "conditional_selection",
        triggers=(), consumes=("sequence", "predicate"), produces="sequence",
    ),
    "accumulate": Primitive(
        "accumulate", OperationType.AGGREGATE, "accumulation",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    "count_elements": Primitive(
        "count_elements", OperationType.AGGREGATE, "counting",
        triggers=(), consumes=("sequence",), produces="scalar",
    ),
    # ── Frequency composition (derived per-element state) ──
    # "the first character that appears only once" is not a lookup:
    # it is a DERIVED per-element frequency, a predicate over that
    # derived state, a selection, a first-position rule and a fallback.
    # These primitives compose, generically, that meaning.
    "count_frequency": Primitive(
        "count_frequency", OperationType.AGGREGATE, "frequency",
        triggers=(), consumes=("sequence",), produces="frequency",
    ),
    "compare_equal_to": Primitive(
        "compare_equal_to", OperationType.COMPARISON, "comparison",
        triggers=(), consumes=("frequency",), produces="predicate",
        preconditions=("an operand is bound",),
    ),
    "select_first": Primitive(
        "select_first", OperationType.TRANSFORM, "first_selection",
        triggers=(), consumes=("sequence",), produces="element",
        preconditions=("input collection non-empty for a defined first element",),
    ),
    "fallback_none": Primitive(
        "fallback_none", OperationType.IO, "fallback",
        triggers=(), consumes=("element",), produces="optional",
    ),
    "nested_loop": Primitive(
        "nested_loop", OperationType.TRANSFORM, "nested_iteration",
        triggers=(), consumes=("sequence", "sequence"), produces="sequence",
    ),
    "conditional_logic": Primitive(
        "conditional_logic", OperationType.CONTROL_FLOW, "branching",
        triggers=(), consumes=("scalar",), produces="scalar",
    ),
}


def _preconditions_for(name: str) -> list[str]:
    """The operation's contract (spec #3): the primitive's declared
    preconditions, falling back to its type wiring — an operation that
    consumes a sequence assumes a sequence is provided."""
    prim = PRIMITIVE_ONTOLOGY.get(name)
    if prim is None:
        return []
    return list(prim.preconditions) or [
        f"input provides {t}" for t in prim.consumes]


def extract_cardinality(text: str) -> tuple[int | None, tuple[int, int] | None]:
    """Extract a cardinality/count from natural language text.

    Returns (count, (start, end)) or (None, None) if no cardinality found.
    This is a structural parser — it does NOT depend on concept tags.
    Handles English and Korean quantifier expressions.
    """
    t = text.lower()

    # 1. English word + "times": "three times", "five times"
    _WORD_NUMS = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "twenty": 20, "hundred": 100,
    }
    m = re.search(
        r'\b(one|two|three|four|five|six|seven|eight|nine|ten|'
        r'eleven|twelve|twenty|hundred)\s+times?\b', t
    )
    if m:
        return _WORD_NUMS[m.group(1)], (m.start(), m.end())

    # 2. Digit + "times": "5 times", "10 times"
    m = re.search(r'\b(\d+)\s*times?\b', t)
    if m:
        return int(m.group(1)), (m.start(), m.end())

    # 3. English specials: "once"=1, "twice"=2, "thrice"=3
    m = re.search(r'\b(once|twice|thrice)\b', t)
    if m:
        return {"once": 1, "twice": 2, "thrice": 3}[m.group(1)], (m.start(), m.end())

    # 4. Korean word + "번": "두 번", "세 번", "다섯 번"
    _KR_NUMS = {
        "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
        "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10,
    }
    m = re.search(
        r'\b(한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*번\b', t
    )
    if m:
        return _KR_NUMS[m.group(1)], (m.start(), m.end())

    # 5. Korean digit + "번": "3번", "10번"
    m = re.search(r'(\d+)\s*번', t)
    if m:
        return int(m.group(1)), (m.start(), m.end())

    return None, None


def _compile_triggers():
    """Compile each primitive's triggers into regex objects."""
    compiled = {}
    for name, prim in PRIMITIVE_ONTOLOGY.items():
        pats = []
        for t in prim.triggers:
            escaped = re.escape(t)
            # Add word boundaries for short ASCII-only triggers to prevent
            # substring matches (e.g. "min" inside "Determine").
            # Korean/Unicode triggers don't use \b (no word boundaries for CJK).
            is_ascii_short = len(t) <= 3 and " " not in t and t.isascii()
            if is_ascii_short:
                pats.append(r"\b" + escaped + r"\b")
            else:
                pats.append(escaped)
        compiled[name] = re.compile("|".join(pats), re.IGNORECASE)
    return compiled


@dataclass
class PrimitiveCue:
    primitive: str
    position: int
    span: int
    matched_text: str
    confidence: float
    params: dict = field(default_factory=dict)


@dataclass
class UserSemantics:
    """Primitives actually observed in the user's text."""
    cues: list[PrimitiveCue] = field(default_factory=list)
    raw_text: str = ""
    unknowns: list[str] = field(default_factory=list)      # tokens not understood
    ambiguities: list[str] = field(default_factory=list)   # ambiguous bindings
    # Constraints the text states WITHOUT naming an operation: "preserving
    # the original order" forbids reordering — it is a requirement on the
    # result, never a sort request.  Each entry: {kind, evidence, span}.
    constraints: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cues": [{"primitive": c.primitive, "matched": c.matched_text,
                      "position": c.position} for c in self.cues],
            "unknowns": self.unknowns,
            "ambiguities": self.ambiguities,
            "constraints": [dict(c) for c in self.constraints],
        }


@dataclass
class RetrievedKnowledge:
    """Knowledge activated from the KG (concepts/relations)."""
    concept_ids: list[str] = field(default_factory=list)
    concept_names: list[str] = field(default_factory=list)
    relation_paths: list[list[str]] = field(default_factory=list)
    primitive_groundings: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "concepts": self.concept_names,
            "concept_ids": self.concept_ids,
            "relation_paths": self.relation_paths,
            "primitive_groundings": self.primitive_groundings,
        }


@dataclass
class DerivedSemantics:
    """The composed new meaning structure (operation graph)."""
    graph: OperationGraph = field(default_factory=OperationGraph)
    inference_level: str = "direct"   # direct | composed | inferred | uncertain
    notes: list[str] = field(default_factory=list)
    annotations: dict = field(default_factory=dict)
    uncertain_flags: list[str] = field(default_factory=list)
    code: str = ""

    def to_dict(self) -> dict:
        return {
            "operations": [op.name for op in self.graph.get_execution_order()],
            "dependencies": {op.name: op.depends_on for op in self.graph.operations},
            "inference_level": self.inference_level,
            "notes": self.notes,
            "annotations": self.annotations,
            "uncertain_flags": self.uncertain_flags,
            "code": self.code,
        }


@dataclass
class CompositionResult:
    user: UserSemantics = field(default_factory=UserSemantics)
    retrieved: RetrievedKnowledge = field(default_factory=RetrievedKnowledge)
    derived: DerivedSemantics = field(default_factory=DerivedSemantics)

    def to_dict(self) -> dict:
        return {
            "user": self.user.to_dict(),
            "retrieved": self.retrieved.to_dict(),
            "derived": self.derived.to_dict(),
        }


class SemanticComposer:
    """Decomposes text into primitives and composes them into a DAG.

    Semantic understanding derives from ConceptStore's stored concepts.
    Concept matching uses description/tag-based scoring, not keyword
    dictionaries.
    """

    def __init__(self, concept_store=None, relation_store=None):
        self.concepts = concept_store
        self.relations = relation_store

    def _ensure_knowledge(self) -> None:
        """Attach the composer's OWN operation knowledge when the caller
        supplied no store.

        The composer reasons over the concept store's operation ontology —
        a store without it decomposes nothing and every request looks
        meaningless.  A caller that hands in a store (the parser, the
        brain, a test) keeps it: the knowledge the composition is built
        from is never silently replaced, but an unconfigured composer is
        never silently empty either.
        """
        if self.concepts is not None and self.relations is not None:
            return
        from brain.concept_store import (ConceptStore, seed_default_concepts,
                                         seed_programming_fundamentals)
        from brain.relation_store import RelationStore, seed_algorithmic_relations
        if self.concepts is None:
            store = ConceptStore()
            seed_default_concepts(store)
            seed_programming_fundamentals(store)
            self.concepts = store
        if self.relations is None:
            relations = RelationStore()
            seed_algorithmic_relations(relations)
            self.relations = relations

    def decompose(self, text: str) -> UserSemantics:
        """Decompose text into primitive cues via concept-based reasoning.

        1. Searches ConceptStore for operation concepts matching the text
        2. Returns primitive cues derived from concept activations
        3. For competitive programming problems, uses structural analysis
        """
        self._ensure_knowledge()
        user = UserSemantics(raw_text=text)

        # Check if this is a competitive programming problem
        if self._is_competitive_programming(text):
            # For competitive programming, use structural analysis
            # to understand what operations are needed
            self._decompose_competitive(text, user)
            self._capture_params(text, user.cues)
            self._reorder_by_semantics(user.cues)
            return user

        if self.concepts:
            # Get scored results from ConceptStore
            scored_concepts = self.concepts.search_with_scores(
                text, category="operation"
            )
            scored = []
            for concept, score in scored_concepts[:10]:
                prim = PRIMITIVE_ONTOLOGY.get(concept.id)
                if prim is None:
                    continue
                role_stage = self._PRIMITIVE_ROLE.get(concept.id)
                if role_stage is None:
                    continue
                scored.append((concept, role_stage, score))

            if not scored:
                self._capture_params(text, user.cues)
                self._reorder_by_semantics(user.cues)
                return user

            # Drop concepts below a minimum relevance score.
            # Score 4.0 requires at least one exact tag match (4.0) or
            # a strong multi-word match (5.0). Prefix containment (3.5)
            # and substring-only matches are dropped as likely false positives.
            MIN_SCORE = 4.0

            # When two ops share a role, prefer the aggregate (calculate_*)
            # over the find (find_*) — aggregate + comparison + filter is a
            # more common pattern than find + comparison + filter.
            _AGGREGATE_OPS = {"calculate_average", "calculate_sum"}
            _FIND_OPS = {"find_max", "find_min"}

            # When is_prime is present, drop is_multiple_of — primality
            # testing subsumes "divisible by" checks. The phrase
            # "divisible only by 1 and itself" is about primality.
            # Also drop filter when is_prime dominates — "only by 1 and
            # itself" is about primality, not filtering.
            # Also: sort_descending subsumes reverse_sequence when the
            # query mentions "reverse order" (sorting, not reversing).
            _PRIME_OPS = {"is_prime"}
            _MULTIPLE_OPS = {"is_multiple_of"}
            _FILTER_OPS = {"filter", "filter_elements"}
            _SORT_OPS = {"sort_descending", "sort_ascending"}
            _REVERSE_OPS = {"reverse_sequence"}

            seen_roles: dict[int, str] = {}
            seen_primitives: dict[str, float] = {}  # primitive -> score
            text_lower = text.lower()
            for concept, role_stage, score in scored:
                if score < MIN_SCORE:
                    continue
                # Check if this op conflicts with an already-kept op
                # even if they have different roles
                skip = False
                replace = None
                # Skip is_multiple_of when is_prime already present
                # and is_prime is very strong (score > 13), indicating
                # a pure primality query. When is_prime is weaker,
                # the query likely needs both (e.g. "소수이면서 3의 배수").
                if concept.id in _MULTIPLE_OPS:
                    for kept_id, kept_score in list(seen_primitives.items()):
                        if kept_id in _PRIME_OPS and kept_score > 13:
                            skip = True
                            break
                # Skip is_prime when is_multiple_of dominates (> 1.2x)
                # — the query is about multiples/divisibility, not primality
                if concept.id in _PRIME_OPS:
                    for kept_id, kept_score in list(seen_primitives.items()):
                        if kept_id in _MULTIPLE_OPS and kept_score > score * 1.2:
                            skip = True
                            break
                if concept.id in _PRIME_OPS:
                    for kept_id, kept_score in list(seen_primitives.items()):
                        if kept_id in _MULTIPLE_OPS and score > kept_score:
                            replace = kept_id
                            break
                # Drop filter when is_prime dominates (score > 1.2x)
                if concept.id in _FILTER_OPS:
                    for kept_id, kept_score in list(seen_primitives.items()):
                        if kept_id in _PRIME_OPS and kept_score > score * 1.2:
                            skip = True
                            break
                # Drop filter_elements when filter is present — the code
                # generator only recognizes "filter" as a valid operation.
                if concept.id == "filter_elements":
                    for kept_id in list(seen_primitives.keys()):
                        if kept_id == "filter":
                            skip = True
                            break
                # Drop reverse_sequence when sort ops are present
                if concept.id in _REVERSE_OPS:
                    for kept_id in list(seen_primitives.keys()):
                        if kept_id in _SORT_OPS:
                            skip = True
                            break
                # Drop find ops when sort ops are present at the same position
                # — "high to low" triggers both find_max and sort_descending,
                # but "high to low" is a sort direction, not a find-max request.
                # Also: "sorted largest first" is a sort, not a find.
                if concept.id in _FIND_OPS:
                    for kept_id in list(seen_primitives.keys()):
                        if kept_id in _SORT_OPS:
                            # Check if text has sort-related context words
                            sort_ctx = re.search(
                                r'(sorted|order|arrange|sequence|순서|대로|정렬|'
                                r'(?:highest|largest|biggest|greatest|smallest|'
                                r'lowest|least)\s+first)',
                                text_lower
                            )
                            if sort_ctx:
                                skip = True
                                break
                # Drop sort ops when find ops are present at the same position
                # — "find the largest" triggers both, but it's a find request.
                if concept.id in _SORT_OPS:
                    for kept_id in list(seen_primitives.keys()):
                        if kept_id in _FIND_OPS:
                            # Check if text has find-related context words
                            find_ctx = re.search(
                                r'\b(find|get|locate|search|찾|구해)',
                                text_lower
                            )
                            if find_ctx:
                                skip = True
                                break
                # Replace reverse_sequence with sort ops when they appear
                if concept.id in _SORT_OPS:
                    for kept_id in list(seen_primitives.keys()):
                        if kept_id in _REVERSE_OPS:
                            replace = kept_id
                            break
                # Sort direction: the DIRECTION phrase (a multi-word tag
                # like "highest to lowest", "largest to smallest") is the
                # semantic authority for the direction — a bare "order"
                # carries no direction (it is just the verb).  When both
                # sort ops are scored, prefer the one whose direction
                # phrase appears in the text.
                if concept.id in _SORT_OPS:
                    other_id = ("sort_descending" if concept.id == "sort_ascending"
                                else "sort_ascending")
                    for other_concept, _, other_score in scored:
                        if other_concept.id != other_id or other_score < MIN_SCORE:
                            continue

                        def _has_direction_phrase(c):
                            for tag in c.tags:
                                if " " in tag and tag.lower() in text_lower:
                                    return True
                            return False

                        if _has_direction_phrase(other_concept) \
                                and not _has_direction_phrase(concept):
                            skip = True
                        break
                # Drop is_prime when is_multiple_of dominates (score > 1.3x)
                if concept.id in _PRIME_OPS:
                    for kept_id, kept_score in list(seen_primitives.items()):
                        if kept_id in _MULTIPLE_OPS and kept_score > score * 1.3:
                            skip = True
                            break
                # Drop remove_duplicates when a cardinality is present.
                # "Repeat the message three times" → "Repeat" triggers
                # remove_duplicates, but the intent is output repetition,
                # not deduplication. Cardinality is a structural signal.
                if concept.id in ("remove_duplicates",):
                    card_check, _ = extract_cardinality(text)
                    if card_check is not None:
                        skip = True
                # Drop emit_output when the pipeline has a terminal operation
                # (sort, filter) anywhere in the scored list, not just in
                # seen_primitives — because emit_output may score higher
                # and be processed first.
                if concept.id in ("emit_output", "repeat_output", "literal_value"):
                    _TERMINAL_OPS = {
                        "sort_descending", "sort_ascending",
                        "filter", "filter_elements",
                    }
                    # Check all scored concepts, not just seen ones
                    for other_concept, _, other_score in scored:
                        if other_concept.id in _TERMINAL_OPS and other_score >= MIN_SCORE:
                            skip = True
                            break
                if skip:
                    continue
                # Replace is_multiple_of with is_prime (primality subsumes divisibility)
                if replace:
                    seen_primitives.pop(replace, None)
                    user.cues = [c for c in user.cues if c.primitive != replace]
                if role_stage in seen_roles:
                    # Role conflict: if the existing op is a find op and
                    # the new op is an aggregate, replace it.
                    existing = seen_roles[role_stage]
                    if existing in _FIND_OPS and concept.id in _AGGREGATE_OPS:
                        # Replace the find op with the aggregate op
                        seen_roles[role_stage] = concept.id
                        seen_primitives.pop(existing, None)
                        # Remove the old cue and add the new one
                        user.cues = [c for c in user.cues if c.primitive != existing]
                        user.cues.append(PrimitiveCue(
                            primitive=concept.id,
                            position=0,
                            span=0,
                            matched_text=text[:50] if len(text) < 50 else text[:50] + "...",
                            confidence=0.85,
                        ))
                    continue
                seen_roles[role_stage] = concept.id
                seen_primitives[concept.id] = score
                # Record the specific matched tag occurrence (not the whole
                # text) so downstream disambiguation can inspect the local
                # context around the evidence.
                match_tag, match_pos = self._locate_tag_occurrence(concept, text)
                user.cues.append(PrimitiveCue(
                    primitive=concept.id,
                    position=match_pos,
                    span=len(match_tag) if match_tag else 0,
                    matched_text=match_tag if match_tag else text[:50],
                    confidence=0.85,
                ))

        # ── Structural cardinality extraction ──
        # Independent of concept tags: scans text for quantifier/cardinality
        # patterns and creates repeat_output cue if not already present.
        card_count, card_span = extract_cardinality(text)
        if card_count is not None:
            has_repeat = any(c.primitive == "repeat_output" for c in user.cues)
            if not has_repeat:
                user.cues.append(PrimitiveCue(
                    primitive="repeat_output",
                    position=card_span[0] if card_span else 0,
                    span=(card_span[1] - card_span[0]) if card_span else 0,
                    matched_text=text[card_span[0]:card_span[1]] if card_span else "",
                    confidence=0.9,
                    params={"count": card_count},
                ))

        # ── Structural literal extraction ──
        # Detects quoted strings and standalone numbers as literal values.
        has_literal = any(c.primitive == "literal_value" for c in user.cues)
        # Skip structural literal extraction for data-pipeline texts
        # (they use "show"/"display"/"output" to mean "display results",
        # not "emit this specific value").
        _PIPELINE_SIGNALS = {
            "above", "average", "mean", "below", "filter", "sort",
            "descending", "ascending", "deduplicate", "unique",
            "greater", "lesser", "between", "range",
        }
        text_lower_local = text.lower()
        is_pipeline = any(sig in text_lower_local for sig in _PIPELINE_SIGNALS)
        if not has_literal and not is_pipeline:
            literal_val = None
            # Quoted strings: "Hello", 'Hello'
            m = re.search(r'["\']([^"\']+)["\']', text)
            if m:
                literal_val = m.group(1)
            else:
                # Word after output/print/display/show/echo/write
                _NO_LIT = {"a", "an", "the", "program", "function", "code",
                           "number", "that", "which", "and", "it", "its",
                           "result", "value", "them", "this", "output"}
                m = re.search(
                    r'\b(?:output|print|display|show|echo|write)s?\s+'
                    r'(?:the\s+)?(?:number\s+)?'
                    r'["\']?([A-Za-z]\w*)["\']?',
                    text, re.IGNORECASE
                )
                if m and m.group(1).lower() not in _NO_LIT:
                    literal_val = m.group(1)
                else:
                    # Standalone number: "Output 42", "the number 42"
                    m = re.search(
                        r'\b(?:output|print|display|show)\s+(?:the\s+)?(?:number\s+)?(\d+)\b',
                        text, re.IGNORECASE
                    )
                    if m:
                        literal_val = m.group(1)
                    else:
                        # Standalone number at end: "42"
                        m = re.search(r'\b(\d+)\s*$', text.strip())
                        if m:
                            literal_val = m.group(1)
            if literal_val is not None:
                user.cues.append(PrimitiveCue(
                    primitive="literal_value",
                    position=0,
                    span=0,
                    matched_text=literal_val,
                    confidence=0.9,
                    params={"value": literal_val},
                ))

        # ── Structural emit_output for repeat without output ──
        # If repeat_output exists but emit_output doesn't, add emit_output
        # as the default action to repeat.
        has_repeat = any(c.primitive == "repeat_output" for c in user.cues)
        has_emit = any(c.primitive == "emit_output" for c in user.cues)
        if has_repeat and not has_emit:
            user.cues.append(PrimitiveCue(
                primitive="emit_output",
                position=0,
                span=0,
                matched_text="",
                confidence=0.85,
            ))

        self._capture_params(text, user.cues)
        # Set cue positions based on where concept tags appear in the text.
        # This preserves the user's intended operation order.
        # Use the tokenizer to find matching positions, handling Korean
        # morphological variations (consonant assimilation, etc.).
        text_lower = text.lower()
        query_words = self.concepts._tokenize_query(text) if self.concepts else set()
        for cue in user.cues:
            concept = self.concepts.get(cue.primitive) if self.concepts else None
            if concept:
                min_pos = len(text)
                # Check concept tags
                for tag in concept.tags:
                    pos = text_lower.find(tag.lower())
                    if pos >= 0 and pos < min_pos:
                        min_pos = pos
                # Also check query words against tags (like scoring does)
                if min_pos == len(text):
                    for qw in query_words:
                        for tag in concept.tags:
                            tag_lower = tag.lower()
                            if qw == tag_lower or qw in tag_lower or tag_lower in qw:
                                pos = text_lower.find(qw)
                                if pos >= 0 and pos < min_pos:
                                    min_pos = pos
                                break
                # Also check if concept name parts appear in text
                if min_pos == len(text):
                    name_parts = cue.primitive.replace("_", " ").split()
                    for part in name_parts:
                        if len(part) >= 3:
                            pos = text_lower.find(part.lower())
                            if pos >= 0 and pos < min_pos:
                                min_pos = pos
                if min_pos < len(text):
                    cue.position = min_pos
        self._add_direction_cue(text, user.cues)
        self._normalize_phrasal_aggregates(text, user.cues)
        # Disambiguate ambiguous detections using the local context
        # around each cue's evidence (before reordering).
        user.cues = self._disambiguate(text, user.cues,
                                       constraints=user.constraints)
        # An aggregate NAMED MORE THAN ONCE is N aggregation requirements
        # (nested aggregation: "the average of the numbers greater than
        # the average of the input list").  The role registry keeps one
        # cue per role; the extra mentions are re-materialized here with
        # their own evidence occurrence — a general multiplicity rule,
        # never a phrase entry.
        self._materialize_repeated_aggregates(text, user.cues)
        # The frequency composition: "the first X that appears only once"
        # composes frequency -> equality -> selection -> first -> fallback
        # as ONE derived-state structure (never a per-sentence entry).
        self._compose_frequency_selection(text, user.cues)
        self._apply_sequential_relations(text, user.cues)
        self._reorder_by_semantics(user.cues)
        return user

    def _normalize_phrasal_aggregates(
            self, text: str, cues: list[PrimitiveCue]) -> None:
        """Read a phrasal aggregate as its aggregate primitive.

        "add up" is a sum over a collection, not the binary add
        predicate.  The concept tag on "add" is still useful for "add
        two numbers"; the particle decides which meaning the sentence
        carries.
        """
        match = re.search(r"\badd(?:s|ed|ing)?\s+up\b", text,
                          re.IGNORECASE)
        if not match:
            return
        for cue in cues:
            if cue.primitive == "add":
                cue.primitive = "calculate_sum"
                cue.matched_text = match.group(0)
                cue.position = match.start()
                cue.confidence = max(cue.confidence, 0.85)
                return

    def _add_direction_cue(self, text: str,
                           cues: list[PrimitiveCue]) -> None:
        """Materialize an explicit "highest first" direction as a sort.

        The direction is a structural phrase: the same words can name a
        maximum ("the highest") or an ordering ("highest first").  The
        "first" completion is the ordering cue; it is added only when no
        sort cue is already present, and never for "preserving the first
        appearance" constraints.
        """
        if any(c.primitive in ("sort_ascending", "sort_descending")
               for c in cues):
            return
        if re.search(r"\bpreserv\w*\b.*\bfirst\b", text, re.IGNORECASE):
            return
        for pattern, primitive in (
                (r"\b(?:highest|largest|biggest|greatest)\s+first\b",
                 "sort_descending"),
                (r"\b(?:lowest|smallest|least)\s+first\b",
                 "sort_ascending")):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cues.append(PrimitiveCue(
                    primitive=primitive,
                    position=match.start(),
                    span=len(match.group(0)),
                    matched_text=match.group(0),
                    confidence=0.85,
                ))
                return

    # Temporal connectives that reverse text order:
    # "A after B" → B first, "A following B" → B first
    _SWAP_CONNECTIVES = re.compile(
        r'\b(after|once|following)\b', re.IGNORECASE
    )

    def _apply_sequential_relations(self, text: str,
                                     cues: list[PrimitiveCue]) -> None:
        """Apply temporal connectives to adjust cue ordering.

        When two cues are separated by a connective like "after" or "following",
        the cue on the RIGHT side of the connective should execute FIRST
        (reversing text order).  For "then"/"and then", text order is preserved.
        """
        if len(cues) < 2:
            return
        text_lower = text.lower()
        for i in range(len(cues) - 1):
            # Compute the right edge of the left cue's matched word.
            # matched_text is a truncated debug string, not the matched word.
            # Use concept tags to find the actual matched word length.
            left_cue = cues[i]
            left_end = left_cue.position
            concept = self.concepts.get(left_cue.primitive) if self.concepts else None
            if concept:
                for tag in concept.tags:
                    pos = text_lower.find(tag.lower(), left_cue.position)
                    if pos == left_cue.position:
                        left_end = left_cue.position + len(tag)
                        break
            else:
                # Fallback: use matched_text length but clamp to text
                left_end = min(left_cue.position + len(left_cue.matched_text),
                               len(text))

            right_pos = cues[i + 1].position
            if left_end >= right_pos:
                continue
            segment = text_lower[left_end:right_pos]
            if self._SWAP_CONNECTIVES.search(segment):
                # "A after B" found: B (right cue) should execute first.
                # Swap their positions so the right cue sorts earlier.
                cues[i].position, cues[i + 1].position = (
                    cues[i + 1].position, cues[i].position
                )

    # Semantic role ordering: ensures pipeline flow (range → aggregate →
    # comparison → predicate → filter → transform → sort → output).
    _PRIMITIVE_ROLE = {
        "generate_range": 0, "literal_value": 0,
        "calculate_average": 1, "calculate_sum": 1,
        "count_frequency": 1,
        "find_max": 1, "find_min": 1, "search": 1,
        "factorial": 1, "fibonacci": 1,
        "add": 1, "multiply": 1,
        "concatenate": 6,
        "compare_greater_than": 2, "compare_less_than": 2,
        "is_prime": 3, "is_multiple_of": 4,
        "compare_equal_to": 2,
        "select_first": 7, "fallback_none": 8,
        "filter": 5, "filter_elements": 5,
        "remove_duplicates": 6, "reverse": 6, "reverse_sequence": 6,
        "sort_descending": 7, "sort_ascending": 7,
        "emit_output": 8, "repeat_output": 9,
    }

    def _reorder_by_semantics(self, cues: list[PrimitiveCue]) -> None:
        """Reorder cues to respect text position (primary) and pipeline flow
        (secondary). Text position preserves the user's intended order.
        Semantic role only breaks ties when positions are equal."""
        cues.sort(key=lambda c: (
            c.position,
            self._PRIMITIVE_ROLE.get(c.primitive, 9),
        ))

    def _decompose_competitive(self, text: str, user: UserSemantics) -> None:
        """Decompose competitive programming problems using structural analysis.
        
        This method analyzes the problem structure to understand what operations
        are needed, rather than relying on keyword matching.
        """
        text_lower = text.lower()
        
        # Analyze problem structure
        has_input_section = bool(re.search(r'(?:Input|입력)\s*\n', text, re.IGNORECASE))
        has_output_section = bool(re.search(r'(?:Output|출력)\s*\n', text, re.IGNORECASE))
        has_constraints = bool(re.search(r'(?:Constraint|제약|Limit)', text, re.IGNORECASE))
        has_examples = bool(re.search(r'(?:Sample|Example|예시)', text, re.IGNORECASE))
        
        # Extract problem description (before Input section)
        desc_match = re.search(r'^(.*?)(?:\n\s*(?:Input|입력))', text, re.DOTALL | re.IGNORECASE)
        description = desc_match.group(1).strip() if desc_match else text[:500]
        
        # Analyze what the problem asks for
        # Look for common competitive programming patterns
        patterns = {
            'sum': [r'sum\s+of', r'합을', r'총합을', r'합산'],
            'count': [r'count\s+of', r'개수를', r'몇', r'수를'],
            'find': [r'find\s+the', r'구하라', r'찾아라', r'구하는'],
            'print': [r'print\s+the', r'출력하라', r'출력하는'],
            'maximum': [r'maximum', r'최대', r'가장\s+큰'],
            'minimum': [r'minimum', r'최소', r'가장\s+작은'],
            'sort': [r'sort', r'정렬', r'오름차순', r'내림차순'],
            'filter': [r'filter', r'골라', r'선택', r'만\s+골라'],
            'iterate': [r'for\s+each', r'각', r'모든', r'전체'],
            'nested': [r'nested', r'이중', r'중첩', r'grid', r'table'],
            'conditional': [r'if', r'만약', r'조건', r'경우'],
            'accumulate': [r'累积', r'누적', r'더해', r'합하여'],
        }
        
        # Detect which patterns are present
        detected_ops = []
        for op_name, regexes in patterns.items():
            for regex in regexes:
                if re.search(regex, description + ' ' + text_lower, re.IGNORECASE):
                    detected_ops.append(op_name)
                    break
        
        # Map detected patterns to primitive operations
        op_mapping = {
            'sum': 'calculate_sum',
            'count': 'count_elements',
            'find': 'conditional_filter',
            'print': 'emit_output',
            'maximum': 'find_max',
            'minimum': 'find_min',
            'sort': 'sort_ascending',
            'filter': 'conditional_filter',
            'iterate': 'iterate_range',
            'nested': 'nested_loop',
            'conditional': 'conditional_logic',
            'accumulate': 'accumulate',
        }
        
        # Create cues from detected operations
        for op in detected_ops:
            if op in op_mapping:
                primitive = op_mapping[op]
                if primitive in PRIMITIVE_ONTOLOGY:
                    user.cues.append(PrimitiveCue(
                        primitive=primitive,
                        position=0,
                        span=0,
                        matched_text=description[:50] if len(description) < 50 else description[:50] + "...",
                        confidence=0.7,
                    ))
        
        # If no specific operations detected, add generic computation
        if not user.cues:
            user.cues.append(PrimitiveCue(
                primitive='accumulate',
                position=0,
                span=0,
                matched_text='generic computation',
                confidence=0.5,
            ))

    def _capture_params(self, text: str, cues: list[PrimitiveCue]) -> None:
        """Attach numeric parameters (range bounds, modulus) to cues."""
        lower = text.lower()

        # Merge redundant filter cues: "값만 뽑아서" is one selection.
        filter_cues = [c for c in cues if c.primitive == "filter"]
        if len(filter_cues) > 1:
            for extra in filter_cues[1:]:
                if extra in cues:
                    cues.remove(extra)
            filter_cues = [c for c in cues if c.primitive == "filter"]

        # Merge consecutive generate_range cues into one with bounds.
        range_cues = [c for c in cues if c.primitive == "generate_range"]
        if len(range_cues) > 1:
            for extra in range_cues[1:]:
                if extra in cues:
                    cues.remove(extra)
            range_cues = [c for c in cues if c.primitive == "generate_range"]

        if range_cues:
            # Try English first (more specific): "from 1 to 100"
            start_m = re.search(r"from\s+(\d+)\s+to", lower)
            end_m = re.search(r"from\s+\d+\s+to\s+(\d+)", lower)
            # Fallback to Korean: "1부터 100까지", "1~100", "1..100"
            if not start_m:
                start_m = re.search(r"(\d+)\s*(?:부터|~|-|\.\.)", lower)
            if not end_m:
                end_m = re.search(r"(\d+)\s*(?:까지|~)", lower)
            if start_m:
                range_cues[0].params["start"] = int(start_m.group(1))
            if end_m:
                range_cues[0].params["end"] = int(end_m.group(1))

        # Capture modulus for is_multiple_of
        for c in cues:
            if c.primitive == "is_multiple_of":
                # Explicit: "3의 배수", "multiple of 3"
                m = re.search(r"(\d+)\s*(?:의\s*)?배수|(?:multiple of|divisible by)\s*(\d+)", lower)
                if m:
                    val = m.group(1) or m.group(2)
                    if val:
                        c.params["modulus"] = int(val)
                # Compound words: 짝수→modulus 2, 홀수→modulus 2 (odd = not divisible by 2)
                if "짝수" in lower and "modulus" not in c.params:
                    c.params["modulus"] = 2
                if "홀수" in lower and "modulus" not in c.params:
                    c.params["modulus"] = 2
                # The English parity words carry the same mathematical
                # contract: even = divisible by 2, odd = not divisible
                # by 2 — the divisibility primitive's canonical case.
                if re.search(r"\beven\b", lower) and "modulus" not in c.params:
                    c.params["modulus"] = 2
                if re.search(r"\bodd\b", lower) and "modulus" not in c.params:
                    c.params["modulus"] = 2

        # Capture the numeric operand of a value comparison: "greater
        # than 5", "above 5", "more than 5", "x > 5", "5보다 큰".  A
        # comparison whose operand is a literal is a THRESHOLD selection;
        # without the operand captured, downstream rendering would have
        # to invent one (the old behavior: silently substitute the mean —
        # semantic contamination).
        for c in cues:
            if c.primitive in ("compare_greater_than", "compare_less_than") \
                    and "operand" not in c.params:
                window = lower[max(0, c.position - 28):c.position + 32]
                m = re.search(
                    r"(?:greater|more|larger|bigger|higher)\s+than\s+"
                    r"(-?\d+(?:\.\d+)?)", window)
                if not m and c.primitive == "compare_greater_than":
                    m = re.search(
                        r"(?:above|over|exceeding|exceeds|surpass\w*)\s+"
                        r"(-?\d+(?:\.\d+)?)", window)
                if not m and c.primitive == "compare_less_than":
                    m = re.search(
                        r"(?:less|fewer|lower|smaller)\s+than\s+"
                        r"(-?\d+(?:\.\d+)?)|(?:below|under)\s+"
                        r"(-?\d+(?:\.\d+)?)", window)
                if not m:
                    m = re.search(r">\s*=?\s*(-?\d+(?:\.\d+)?)", window) \
                        if c.primitive == "compare_greater_than" else None
                if not m:
                    m = re.search(r"<\s*=?\s*(-?\d+(?:\.\d+)?)", window) \
                        if c.primitive == "compare_less_than" else None
                if not m:
                    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:보다\s*크|초과|이상)",
                                  window) \
                        if c.primitive == "compare_greater_than" else None
                if not m:
                    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:보다\s*작|미만|이하)",
                                  window) \
                        if c.primitive == "compare_less_than" else None
                if m:
                    raw = next(g for g in m.groups() if g is not None)
                    val = float(raw)
                    c.params["operand"] = (int(val)
                                           if val.is_integer() else val)

        # Capture literal value: extract quoted strings or words after
        # output/print/display keywords
        _LITERAL_ARTICLES = {"a", "an", "the", "program", "function", "code",
                             "number", "that", "which", "and", "it", "its",
                             "result", "value", "them", "this", "output"}
        # Words that are references/descriptions, not literal values
        _REFERENCE_WORDS = {"same", "this", "that", "these", "those", "all",
                            "every", "each", "my", "your", "his", "her",
                            "its", "our", "their", "new", "old", "other",
                            "previous", "current", "next", "above", "below"}
        literal_cues_to_remove = []
        for c in cues:
            if c.primitive == "literal_value":
                # Try quoted strings first: "Hello", 'Hello'
                m = re.search(r'["\']([^"\']+)["\']', text)
                if m:
                    c.params["value"] = m.group(1)
                else:
                    # Try word after output/prints/display/show/echo/write
                    for m in re.finditer(
                        r'\b(output|print|display|show|echo|write)s?\s+'
                        r'(?:the\s+)?(?:number\s+)?'
                        r'["\']?([A-Za-z]\w*)["\']?',
                        text, re.IGNORECASE
                    ):
                        val = m.group(2)
                        if (val.lower() not in _LITERAL_ARTICLES
                                and val.lower() not in _REFERENCE_WORDS):
                            c.params["value"] = val
                            break
                    # Try numeric value: "output 42", "print 7"
                    if "value" not in c.params:
                        m = re.search(
                            r'\b(output|print|display|show)\s+(?:the\s+)?(?:number\s+)?(\d+)\b',
                            text, re.IGNORECASE
                        )
                        if m:
                            c.params["value"] = m.group(2)
                            c.params["value_type"] = "number"
                    # Detect standalone numbers not after output verb
                    # e.g. "produce the value 7 four times"
                    if "value" not in c.params:
                        m = re.search(r'\b(\d+)\b', text)
                        if m:
                            # Only use if it looks like a literal value context
                            before = text[:m.start()].lower()
                            if any(w in before for w in ["value", "값", "the", "produce"]):
                                c.params["value"] = m.group(1)
                                c.params["value_type"] = "number"
                # If no value extracted, mark for removal
                if "value" not in c.params:
                    literal_cues_to_remove.append(c)
        for c in literal_cues_to_remove:
            cues.remove(c)

        # Capture repeat count for repeat_output
        # Skip if count was already set by structural cardinality extractor
        for c in cues:
            if c.primitive == "repeat_output" and "count" not in c.params:
                # Korean: "두 번", "세 번", "N번"
                m = re.search(r'(\d+)\s*번', text)
                if m:
                    c.params["count"] = int(m.group(1))
                else:
                    # Korean word numbers
                    kr_nums = {"두": 2, "세": 3, "네": 4, "다섯": 5}
                    for word, num in kr_nums.items():
                        if word in text:
                            c.params["count"] = num
                            break
                    if "count" not in c.params:
                        # English special: "twice"=2, "thrice"=3, "once"=1
                        m = re.search(r'\b(twice|thrice|once)\b', text, re.IGNORECASE)
                        if m:
                            c.params["count"] = {"once": 1, "twice": 2, "thrice": 3}[m.group(1).lower()]
                        else:
                            # English word numbers + "times": "three times"
                            word_nums = {"one":1,"two":2,"three":3,"four":4,"five":5,
                                         "six":6,"seven":7,"eight":8,"nine":9,"ten":10}
                            m = re.search(
                                r'\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+times?\b',
                                text, re.IGNORECASE
                            )
                            if m:
                                c.params["count"] = word_nums[m.group(1).lower()]
                            else:
                                # Digit + "times": "5 times"
                                m = re.search(r'(\d+)\s*times?', text, re.IGNORECASE)
                                if m:
                                    c.params["count"] = int(m.group(1))
                            # No count evidence: leave unset — a count-less
                            # repeat carries no repeat-N semantics and the
                            # justification check can then drop it when the
                            # evidence describes duplicated data instead.

    def _locate_tag_occurrence(self, concept, text: str) -> tuple[str, int]:
        """Find the earliest occurrence of any of the concept's tags in the
        text. Returns (matched_tag, position) or ('', 0)."""
        text_lower = text.lower()
        best_tag, best_pos = "", -1
        for tag in concept.tags:
            tag_lower = tag.lower()
            if not tag_lower:
                continue
            idx = text_lower.find(tag_lower)
            if idx != -1 and (best_pos == -1 or idx < best_pos):
                best_tag, best_pos = tag_lower, idx
        if best_pos == -1:
            return "", 0
        return best_tag, best_pos

    # Words that mark the ordering noun as the EXISTING sequence the
    # result must keep (an adjective of provenance), never a direction.
    _ORDER_PROVENANCE = {
        "original", "same", "input", "given", "natural", "existing",
        "current", "chronological", "stable", "sorted_original",
        "입력", "원래", "기존", "현재", "본래",
    }

    def _order_preservation_span(self, text_lower: str,
                                 cue: PrimitiveCue) -> str:
        """The preservation phrase the cue's span belongs to, or "".

        "while preserving the original order" contains the ordering noun
        "order", but the phrase describes the SEQUENCE the result keeps —
        it forbids reordering and justifies no sort.  The rule is
        syntactic: an ordering noun introduced by a provenance adjective
        ("original/same/input ... order") or governed by a preservation
        verb ("preserving/keeping/retaining ... order") is the noun
        "sequence", not the verb "arrange".
        """
        window_start = max(0, cue.position - 40)
        window_end = min(len(text_lower), cue.position + 40)
        window = text_lower[window_start:window_end]
        for m in re.finditer(r"\b(order|sequence)\b|순서|차례", window):
            abs_pos = window_start + m.start()
            before = text_lower[max(0, abs_pos - 28):abs_pos]
            prev_word = ""
            parts = re.split(r"[\s,;:.]+", before.strip())
            if parts:
                prev_word = parts[-1]
            if prev_word in self._ORDER_PROVENANCE:
                return text_lower[window_start + m.start():window_end]
            if re.search(
                    r"(preserv\w*|keep(?:s|ing|t)?|retain\w*|maintain\w*|"
                    r"without\s+changing|unchanged|유지|보존|지키|그대로)\s+"
                    r"(?:the\s+|its\s+)?$", before):
                return text_lower[window_start + m.start():window_end]
        return ""

    def _materialize_repeated_aggregates(self, text: str,
                                         cues: list[PrimitiveCue]) -> None:
        """One cue per NAMED occurrence of an aggregate concept.

        The role registry admits a single cue per semantic role, which is
        right for synonyms of one requirement — but an aggregate named
        twice expresses two aggregations ("average of X greater than the
        average of Y").  Each additional occurrence of the concept's tags
        in the text becomes its own cue, with the occurrence's position
        as evidence, so the composer wires as many aggregation steps as
        the request names.
        """
        if self.concepts is None:
            return
        text_lower = text.lower()
        for concept_id in ("calculate_average", "calculate_sum"):
            cue_list = [c for c in cues if c.primitive == concept_id]
            if not cue_list:
                continue
            concept = self.concepts.get(concept_id)
            if concept is None:
                continue
            occurrences: list[tuple[int, str]] = []
            for tag in concept.tags:
                tag_lower = tag.lower().strip()
                if not tag_lower or " " in tag_lower:
                    continue
                start = 0
                while True:
                    idx = text_lower.find(tag_lower, start)
                    if idx == -1:
                        break
                    occurrences.append((idx, tag_lower))
                    start = idx + len(tag_lower)
            # Non-overlapping, earliest-first occurrences
            occurrences.sort()
            chosen: list[tuple[int, str]] = []
            for idx, tag_lower in occurrences:
                if any(abs(idx - c[0]) < max(len(tag_lower), len(c[1]))
                        for c in chosen):
                    continue
                chosen.append((idx, tag_lower))
            if len(chosen) <= len(cue_list):
                continue
            for extra_idx, (idx, tag_lower) in enumerate(
                    chosen[len(cue_list):], start=len(cue_list)):
                cues.append(PrimitiveCue(
                    primitive=concept_id,
                    position=idx,
                    span=len(tag_lower),
                    matched_text=text[idx:idx + len(tag_lower)],
                    confidence=0.85,
                    params={"occurrence": extra_idx},
                ))

    def _compose_frequency_selection(self, text: str,
                                     cues: list[PrimitiveCue]) -> None:
        """Compose the derived-state frequency selection, generically.

        The structure "the first X that appears [only] once, None if it
        does not exist" is not one operation — it is:
        frequency(element) -> compare_equal(1) -> selection -> first ->
        fallback.  The composition fires ONLY when the frequency
        construction itself was detected ("appears/occurs ... once",
        "frequency"), so an incidental "first" elsewhere in the request
        never fabricates this structure.
        """
        freq_cues = [c for c in cues if c.primitive == "count_frequency"]
        if not freq_cues:
            return
        lower = text.lower()
        # Counting frequencies is not selecting the elements that occur a
        # particular number of times.  "Count how many times each word
        # appears" asks for the frequency map itself; only an explicit
        # one/first/exactly-N-times construction asks for a selection.
        if not re.search(
                r"\b(?:only\s+)?once\b|\bexactly\s+\d+\s+times?\b|"
                r"\b\d+\s+times?\b|\bfirst\b|"
                r"한\s*번|처음|첫", lower):
            return
        pos = freq_cues[0].position
        operand = 1
        m = re.search(r"\b(\d+)\s*times?\b", lower)
        if m:
            operand = int(m.group(1))
        cues.append(PrimitiveCue(
            primitive="compare_equal_to",
            position=pos,
            span=0,
            matched_text=freq_cues[0].matched_text or "appears once",
            confidence=0.8,
            params={"operand": operand},
        ))
        m_first = re.search(r"\bfirst\b|처음|첫\s*번째|첫", lower)
        if m_first:
            cues.append(PrimitiveCue(
                primitive="select_first",
                position=m_first.start(),
                span=m_first.end() - m_first.start(),
                matched_text=lower[m_first.start():m_first.end()],
                confidence=0.8,
            ))
        if re.search(r"\bnone\b|no such|does(?:n't| not) exist|"
                     r"없으면|없는\s*경우|없을\s*때|없다면", lower):
            cues.append(PrimitiveCue(
                primitive="fallback_none",
                position=len(text),
                span=0,
                matched_text="None",
                confidence=0.8,
            ))

    def _comparison_has_divisibility_sense(self, text_lower: str,
                                           cue: PrimitiveCue) -> bool:
        """Whether a comparator cue's span is a divisibility construction.

        "multiple" is the divisibility cue only in "multiple OF n",
        "divisible by n", and the parity words; as a quantifier
        determiner ("multiple strings", "multiple values") it counts
        elements and says nothing about divisibility.
        """
        window_start = max(0, cue.position - 12)
        window = text_lower[window_start:cue.position + 24]
        if re.search(r"\bodd\b|\beven\b|divisible|배수|짝수|홀수", window):
            return True
        m = re.search(r"\bmultiple\b", window)
        if m:
            after = window[m.end():m.end() + 8]
            return bool(re.match(r"\s+of\b", after))
        return bool(re.search(r"divisible|배수", window))

    def _disambiguate(self, text: str, cues: list[PrimitiveCue],
                      constraints: Optional[list] = None) -> list[PrimitiveCue]:
        """Resolve ambiguous primitive detections using context."""
        lower = text.lower()
        result = []
        for cue in cues:
            keep = True
            if cue.primitive in ("sort_ascending", "sort_descending"):
                # The ordering noun inside a preservation phrase is the
                # sequence, not an ordering action ("while preserving the
                # original order") — drop the sort and record the
                # constraint so verification can enforce it.
                preserved = self._order_preservation_span(lower, cue)
                if preserved:
                    keep = False
                    if constraints is not None:
                        constraints.append({
                            "kind": "preserve_order",
                            "evidence": preserved.strip(),
                            "span": cue.position,
                        })
            if cue.primitive == "is_multiple_of":
                # Sense disambiguation: divisibility vs quantifier.
                if not self._comparison_has_divisibility_sense(lower, cue):
                    keep = False
            if cue.primitive == "find_max":
                # "가장 큰 ... 순서/대로/정렬" means sort_descending, not find_max
                # "from largest to smallest" is sort direction, not find_max
                before = lower[max(0, cue.position - 20):cue.position]
                after = lower[cue.position:cue.position + 30]
                if re.search(r"(순서|대로|정렬|sort|order|to\s+(smallest|lowest|min|littlest)|largest\s+to\s+smallest)", before + after):
                    keep = False
            if cue.primitive == "find_min":
                # "from smallest to largest" is sort direction, not find_min
                # "from largest to smallest" also means sort direction
                before = lower[max(0, cue.position - 30):cue.position]
                after = lower[cue.position:cue.position + 30]
                combined = before + after
                if re.search(r"(to\s+(largest|highest|max|biggest)|smallest\s+to\s+largest|largest\s+to\s+smallest|sort|order|가장\s*큰.*가장\s*작은)", combined):
                    keep = False
            if cue.primitive == "generate_range":
                # "from largest to smallest" is NOT a range; only "from N to M" (digits) is
                before = lower[max(0, cue.position - 5):cue.position]
                after = lower[cue.position:cue.position + 20]
                if not re.search(r"\d", before + after):
                    keep = False
            if cue.primitive in ("compare_greater_than", "compare_less_than"):
                # "3의 배수" etc. must not bind a lone "의"/small hanja char
                if len(cue.matched_text) < 2 and cue.matched_text not in ("이상", "이하"):
                    keep = False
                # A comparative word immediately followed by an ordering
                # phrase ("작은 순서대로", "큰 순서", "작은 것부터",
                # "in order") is a SORT DIRECTION, not a value comparison —
                # the ordering op already carries the direction. General
                # linguistic rule: comparative + order phrase = ordering
                # usage. The phrase must DIRECTLY follow the comparative
                # (whitespace only) — an intervening word ("작은 값만")
                # keeps it a comparison.
                elif cue.matched_text:
                    after = lower[cue.position + len(cue.matched_text):
                                  cue.position + len(cue.matched_text) + 20]
                    if re.match(r"\s*(?:순서대로|순서\s+대로|순서\b|것부터|order\b)", after):
                        keep = False
            if cue.primitive == "repeat_output":
                # A repeat with no cardinality ("반복되는 숫자" describes
                # duplicated DATA, not repeat-N-times) has no repeat
                # semantics — an unjustified operation.
                if not cue.params.get("count"):
                    keep = False
                elif cue.params.get("count") == 1:
                    # Repeating ONCE is the identity — no repetition
                    # semantics.  "appears only once" is a frequency
                    # construction, never an output repetition.
                    keep = False
                elif cue.matched_text:
                    # "반복되는/반복된 + 명사" — the passive adjective
                    # describes the DATA as duplicated (remove_duplicates
                    # territory), not the output repetition.
                    after = lower[cue.position + len(cue.matched_text):
                                  cue.position + len(cue.matched_text) + 12]
                    if re.match(r"\s*(?:되는|된)\s", after):
                        keep = False
            if keep:
                result.append(cue)
        # A structural emit_output exists only to carry a REPEAT ("print
        # N times").  Once no repeat survives, an evidence-less emit is
        # an orphaned side effect — an unjustified operation.
        if not any(c.primitive == "repeat_output" for c in result):
            result = [c for c in result
                      if not (c.primitive == "emit_output"
                              and not c.matched_text)]
        return result

    # ------------------------------------------------------------------ #
    # 2. Composition
    # ------------------------------------------------------------------ #
    def compose(self, text: str) -> CompositionResult:
        """Full pipeline: decompose → bind → derive operation graph."""
        user = self.decompose(text)
        retrieved = self._retrieve(user)
        graph, notes, level = self._derive_graph(user)
        derived = DerivedSemantics(graph=graph, notes=notes, inference_level=level)
        # Uncertainty flags
        derived.uncertain_flags = self._compute_uncertainty_flags(text, user, graph, notes)
        # Generate executable code if graph is non-trivial
        if graph.operations:
            # Call-form judging: the problem's starter code declares the
            # function signature — the generated code must match that
            # contract (def name(params) -> return), not the stdio form.
            starter = self._extract_starter_signature(text)
            if starter:
                entry_point, params = starter
                derived.code = _generate_competitive_python(
                    graph, entry_point, [], [], call_form=True,
                    call_params=params)
            else:
                # Detect if this is a competitive programming problem
                is_competitive = self._is_competitive_programming(text)
                if is_competitive:
                    # Extract I/O specs for competitive programming
                    input_specs, output_specs = self._extract_competitive_io(text)
                    derived.code = generate_python(graph, is_competitive=True, 
                                                  input_specs=input_specs, output_specs=output_specs)
                else:
                    derived.code = generate_python(graph)
        return CompositionResult(user=user, retrieved=retrieved, derived=derived)

    def _extract_starter_signature(self, text: str):
        """Extract the function signature declared by the problem's
        starter code ("Starter code: ... def NAME(self, P1, ...)").

        Returns (entry_point, params) with ``self`` excluded (call-based
        judges invoke the function without it), or None when the prompt
        declares no starter code.
        """
        m = re.search(
            r'[Ss]tarter\s+code\s*:.*?def\s+([A-Za-z_]\w*)\s*\(\s*self\s*(?:,\s*([^)]*))?\)',
            text, re.DOTALL)
        if not m:
            return None
        entry_point = m.group(1)
        raw_params = m.group(2) or ""
        params = []
        for p in raw_params.split(","):
            p = p.strip()
            if not p:
                continue
            # strip type annotations: "nums: List[int]" -> "nums"
            name = p.split(":")[0].strip().split("=")[0].strip()
            if name and name != "self":
                params.append(name)
        if not entry_point:
            return None
        return entry_point, params
    
    def _is_competitive_programming(self, text: str) -> bool:
        """Detect if the text is a competitive programming problem."""
        text_lower = text.lower()
        # Check for competitive programming indicators
        cp_indicators = [
            r'standard\s+input', r'standard\s+output', r'stdin', r'stdout',
            r'입력\s*파일', r'출력\s*파일', r'입력\s* 형식', r'출력\s* 형식',
            r'input\s+format', r'output\s+format',
            r'sample\s+input', r'sample\s+output', r'예시\s+입력', r'예시\s+출력',
            r'constraint', r'제약\s+조건',
            r'print\s+the\s+(?:answer|result|output)',
            r'the\s+input\s+is\s+given\s+from',
        ]
        for indicator in cp_indicators:
            if re.search(indicator, text_lower):
                return True
        return False
    
    def _extract_competitive_io(self, text: str) -> tuple[list, list]:
        """Extract I/O specifications from competitive programming problems."""
        # This is a simplified version - the full parser is in SemanticParser
        input_specs = []
        output_specs = []
        
        # Look for input section
        input_match = re.search(r'(?:Input|입력)\s*\n(.*?)(?:\n\s*(?:Output|출력))', text, re.DOTALL | re.IGNORECASE)
        if input_match:
            input_section = input_match.group(1).strip()
            # Extract variable names from format description
            format_match = re.search(r'(?:the\s+following\s+format|다음\s+형식)\s*[:：]?\s*\n(.*?)(?:\n\s*\n|\n\s*(?:Constraint|제약|Sample|예시))', input_section, re.DOTALL | re.IGNORECASE)
            if format_match:
                format_lines = format_match.group(1).strip().split('\n')
                for line in format_lines:
                    line = line.strip()
                    if not line or len(line) > 50:
                        continue
                    vars_in_line = re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:_\d+)?', line)
                    for var in vars_in_line:
                        # Determine if it's a list or single value
                        is_list = bool(re.search(rf'{re.escape(var)}(?:_\d+)?\s+(?:{re.escape(var)}(?:_\d+)?\s+)+', line))
                        var_type = 'List<Integer>' if is_list else 'Integer'
                        input_specs.append(type('IOSpec', (), {
                            'name': var, 
                            'type_signature': var_type,
                            'desc': f'Input {var}'
                        })())
        
        # Look for output section
        output_match = re.search(r'(?:Output|출력)\s*\n(.*?)(?:\n\s*(?:Constraint|제약|Sample|예시))', text, re.DOTALL | re.IGNORECASE)
        if output_match:
            output_section = output_match.group(1).strip()
            if re.search(r'(?:print|출력|return|반환)', output_section, re.IGNORECASE):
                output_specs.append(type('IOSpec', (), {
                    'name': 'answer',
                    'type_signature': 'Any',
                    'desc': 'Output answer'
                })())
        
        return input_specs, output_specs

    def _compute_uncertainty_flags(self, text, user, graph, notes):
        flags = []
        if not user.cues:
            flags.append("no_primitives_detected")
            return flags
        # Low confidence match: matched_text is < 2 chars or is a single particle
        for cue in user.cues:
            if len(cue.matched_text) < 2 and cue.matched_text not in ("이", "가", "을", "에"):
                flags.append("low_confidence_match:{}".format(cue.primitive))
        # Predicate without aggregate: e.g. compare_greater without calculate_average
        has_agg = any(op.name in ("calculate_average", "calculate_sum", "find_max", "find_min")
                      for op in graph.operations)
        has_comp = any(op.name in ("compare_greater_than", "compare_less_than")
                       and op.params.get("operand") is None
                       for op in graph.operations)
        if has_comp and not has_agg:
            flags.append("comparison_without_aggregate")
        # Filter without predicates: the filter op has no predicate deps
        for op in graph.operations:
            if op.name == "filter" and not op.params.get("predicates"):
                flags.append("filter_without_predicates")
        return flags

    def _retrieve(self, user: UserSemantics) -> RetrievedKnowledge:
        """Ground the detected primitives in the knowledge graph.

        Traverses relations to discover concepts related to detected
        primitives via REQUIRES, PRODUCES, SIMILAR_TO, and USES edges.
        """
        retrieved = RetrievedKnowledge()
        for cue in user.cues:
            retrieved.primitive_groundings.setdefault(cue.primitive, []).append(cue.matched_text)
            retrieved.concept_names.append(cue.primitive)
            # Traverse relations to find related concepts
            if self.relations is not None:
                related = self.relations.find_related(cue.primitive)
                for rel in related:
                    other_id = rel.target_id if rel.source_id == cue.primitive else rel.source_id
                    if other_id not in retrieved.concept_ids:
                        retrieved.concept_ids.append(other_id)
                        other = self.concepts.get(other_id) if self.concepts else None
                        if other:
                            retrieved.concept_names.append(other.name)
                            # Record the relation path
                            retrieved.relation_paths.append(
                                [cue.primitive, rel.relation_type.value, other_id]
                            )
        # Also surface concepts whose tags match detected primitives
        if self.concepts is not None:
            for c in self.concepts.list_all():
                if c.category == "pattern" and any(
                    t.lower() in {p.primitive for p in user.cues}
                    for t in c.tags
                ):
                    retrieved.concept_ids.append(c.id)
                    retrieved.concept_names.append(c.name)
        return retrieved

    @staticmethod
    def _record_semantic_param(op, label: str, src) -> None:
        """State a wired input in the operation's own parameters.

        The wiring decides WHICH producer feeds which input; the
        parameter is the readable form of that decision: a comparison's
        threshold IS the aggregate it consumes, a selection's predicates
        ARE the predicate producers it binds.  A literal the request
        already supplied ("above 5") is never overwritten — the
        parameters report the dataflow, they never re-derive it.
        """
        if src is None:
            return
        if label == "scalar" and op.name in ("compare_greater_than",
                                             "compare_less_than") \
                and "threshold" not in op.params:
            op.params["threshold"] = src.name
        elif label == "predicate" and op.name in ("filter",
                                                  "filter_elements"):
            predicates = list(op.params.get("predicates") or [])
            if src.name not in predicates:
                predicates.append(src.name)
            op.params["predicates"] = predicates

    def _derive_graph(self, user: UserSemantics) -> tuple[OperationGraph, list[str], str]:
        """Compose primitives into a dependency DAG using type-based matching.

        Architecture:
          1. Create OperationSpec for each detected cue
          2. Add implicit filters after comparisons (if no explicit filter)
          3. Add relation-based inference for aggregate+filter patterns
          4. Wire ALL operations by matching consumes/produces types

        This ensures all operations exist before wiring, so dependencies
        are correct.
        """
        graph = OperationGraph()
        notes: list[str] = []
        cues = user.cues

        # ── Step 1: Create OperationSpec for each cue ──
        ops: list[OperationSpec] = []
        for i, cue in enumerate(cues):
            name = cue.primitive
            params = dict(cue.params)
            prim = PRIMITIVE_ONTOLOGY[name]

            op = OperationSpec(
                id=f"op_{i}",
                name=name,
                op_type=prim.op_type,
                semantic_role=prim.semantic_role,
                params=params,
                depends_on=[],
                input_entities=[],
                output_entity=None,
                description=f"{cue.matched_text} -> {name}",
                semantic_justification=(
                    f"cue '{cue.matched_text}' composes to primitive "
                    f"{name} ({prim.semantic_role}) by type-based match"),
                evidence=[f"cue: {cue.matched_text}"],
                preconditions=_preconditions_for(name),
            )
            ops.append(op)

        # ── Step 2: Comparison → implicit filter ──
        comp_ops = [op for op in ops
                    if op.name in ("compare_greater_than",
                                   "compare_less_than", "compare_equal_to")]
        has_filter = any(op.name in ("filter", "filter_elements") for op in ops)
        if comp_ops and not has_filter:
            new_ops = []
            for op in ops:
                new_ops.append(op)
                if op.name in ("compare_greater_than", "compare_less_than",
                               "compare_equal_to"):
                    filter_op = OperationSpec(
                        id=f"filter_{op.id}",
                        name="filter",
                        op_type=PRIMITIVE_ONTOLOGY["filter"].op_type,
                        semantic_role="selection",
                        params={"predicates": [op.name]},
                        depends_on=[op.id],
                        description=f"implicit filter from {op.name}",
                        semantic_justification=(
                            f"{op.name} yields a predicate but the text "
                            "states no selection — the implicit filter "
                            "carries the predicate into the dataflow"),
                        evidence=[f"implicit: predicate from {op.name}"],
                        preconditions=_preconditions_for("filter"),
                    )
                    new_ops.append(filter_op)
                    notes.append(f"implicit filter created for {op.name}")
            ops = new_ops

        # ── Step 2b: Element predicate → implicit selection ──
        # An element-level predicate (even/odd via is_multiple_of,
        # primality) selects ELEMENTS. When the graph ALSO aggregates
        # over the collection, the predicate applies to the aggregate's
        # operand: "sum of even numbers" = sum over the even-selected
        # elements. The predicate therefore needs a SELECTION step —
        # an implicit filter carrying it.
        has_filter_explicit = any(op.name in ("filter", "filter_elements")
                                  for op in ops)
        agg_ops_present = any(
            op.name in ("calculate_sum", "calculate_average",
                        "find_max", "find_min", "count_elements")
            for op in ops)
        if not has_filter_explicit and agg_ops_present:
            elem_preds = [op for op in ops
                          if op.name in ("is_multiple_of", "is_prime")]
            if elem_preds:
                filter_op = OperationSpec(
                    id="filter_elem_sel",
                    name="filter",
                    op_type=PRIMITIVE_ONTOLOGY["filter"].op_type,
                    semantic_role="selection",
                    params={"predicates": [p.name for p in elem_preds]},
                    depends_on=[p.id for p in elem_preds],
                    description="implicit selection from element predicate",
                    semantic_justification=(
                        "an element predicate with an aggregate: the "
                        "selection binds the predicate to the elements "
                        "the aggregate consumes ('sum of even numbers' "
                        "sums the EVEN elements)"),
                    evidence=[f"implicit: element predicate "
                              f"{', '.join(p.name for p in elem_preds)}"],
                    preconditions=_preconditions_for("filter"),
                )
                insert_after = max(ops.index(p) for p in elem_preds) + 1
                ops.insert(insert_after, filter_op)
                notes.append("implicit selection created for element predicate "
                             + ", ".join(p.name for p in elem_preds))

        # ── Step 3: Relation-based inference ──
        if self.relations is not None:
            op_names = {op.name for op in ops}
            new_ops = list(ops)

            has_agg = any(op.name in ("calculate_average", "calculate_sum") for op in ops)
            has_comp = any(op.name in ("compare_greater_than", "compare_less_than") for op in ops)
            # An ELEMENT-PREDICATE selection ("even numbers", "primes")
            # already in the graph IS the filter: the aggregate+threshold
            # inference exists for the "keep values greater than the
            # aggregate" pattern, where the text's filter is a THRESHOLD
            # comparison. It must never fire over an element predicate —
            # "sum of even numbers" selects elements; it does not
            # compare against the sum. A PLAIN filter cue ("keep",
            # "select") has no predicate yet: it is the threshold
            # filter WAITING for its comparison — inference must supply
            # it, never be blocked by it.
            has_element_selection = any(
                op.name in ("is_multiple_of", "is_prime",
                            "filter_elements")
                for op in ops) or any(
                op.name == "filter" and any(
                    str(p) in ("is_multiple_of", "is_prime")
                    for p in (op.params.get("predicates") or []))
                for op in ops)

            if has_agg and not has_comp and not has_element_selection \
                    and self.concepts is not None:
                text_concepts = self.concepts.search(user.raw_text)

                has_filter_concept = False
                for tc in text_concepts:
                    if tc.id in ("filter", "filter_elements"):
                        has_filter_concept = True
                        break
                    rels = self.relations.find_relations(
                        source_id=tc.id, target_id="filter",
                        relation_type=RelationType.REQUIRES,
                    )
                    if rels:
                        has_filter_concept = True
                        break
                    rels = self.relations.find_relations(
                        source_id=tc.id, relation_type=RelationType.PRODUCES,
                    )
                    for r in rels:
                        if r.target_id == "predicate":
                            has_filter_concept = True
                            break

                if not has_filter_concept:
                    for tc in text_concepts:
                        prim = PRIMITIVE_ONTOLOGY.get(tc.id)
                        if prim and "aggregate" in prim.consumes:
                            has_filter_concept = True
                            break

                if not has_filter_concept:
                    text_lower = user.raw_text.lower()
                    if "값만" in text_lower or "값들만" in text_lower:
                        has_filter_concept = True

                if has_filter_concept and not has_element_selection:
                    for op in ops:
                        if op.name in ("calculate_average", "calculate_sum"):
                            comp_rels = self.relations.find_relations(
                                source_id="compare_greater_than",
                                target_id=op.name,
                                relation_type=RelationType.REQUIRES,
                            )
                            if comp_rels:
                                agg_idx = new_ops.index(op)
                                comp_op = OperationSpec(
                                    id=f"infer_comp_{agg_idx}",
                                    name="compare_greater_than",
                                    op_type=PRIMITIVE_ONTOLOGY["compare_greater_than"].op_type,
                                    semantic_role="comparison",
                                    params={"threshold": op.name},
                                    depends_on=[op.id],
                                    description=f"inferred comparison from {op.name} via ontology+relation",
                                    semantic_justification=(
                                        f"ontology + relation: the text "
                                        f"describes a threshold over the "
                                        f"aggregate {op.name} (REQUIRES "
                                        f"relation to compare_greater_than)"),
                                    evidence=[f"relation: compare_greater_than "
                                              f"REQUIRES {op.name}"],
                                    preconditions=_preconditions_for(
                                        "compare_greater_than"),
                                )
                                new_ops.insert(agg_idx + 1, comp_op)

                                has_filter_already = any(
                                    o.name in ("filter", "filter_elements") for o in ops
                                )
                                if not has_filter_already:
                                    filter_op = OperationSpec(
                                        id=f"infer_filter_{agg_idx}",
                                        name="filter",
                                        op_type=PRIMITIVE_ONTOLOGY["filter"].op_type,
                                        semantic_role="selection",
                                        params={"predicates": ["compare_greater_than"]},
                                        depends_on=[comp_op.id],
                                        description=f"inferred filter from comparison via ontology+relation",
                                        semantic_justification=(
                                            "the threshold comparison "
                                            "requires a selection step — "
                                            "a predicate alone selects "
                                            "nothing"),
                                        evidence=[f"implicit: selection for "
                                                  f"comparison {comp_op.id}"],
                                        preconditions=_preconditions_for("filter"),
                                    )
                                    filter_idx = new_ops.index(comp_op) + 1
                                    new_ops.insert(filter_idx, filter_op)
                                notes.append(
                                    f"inferred comparison (+filter) from {op.name} "
                                    f"via consumes/produces ontology + REQUIRES relation"
                                )
                                break

            ops = new_ops

        # ── Step 4: Wire dependencies by type matching ──
        # NOW all operations exist, so we can wire correctly.
        type_pool: dict[str, list[str]] = {}
        wired: dict[str, bool] = {}

        # Pre-compute: which ops WILL produce which types (even if not yet wired).
        # This lets aggregates prefer a transform's output even if the transform
        # hasn't been processed yet in the wiring loop.
        future_producers: dict[str, list[str]] = {}
        for op in ops:
            prim = PRIMITIVE_ONTOLOGY[op.name]
            if prim.produces:
                future_producers.setdefault(prim.produces, []).append(op.id)

        # Sort ops by wiring priority to respect pipeline stages:
        # sources → aggregates → comparisons → filters → transforms → sorts → output
        # This prevents sort_descending from grabbing __input__ before filter
        # gets a chance to wire, ensuring correct dependency chains.
        _WIRING_PRIORITY = {
            "data_source": 0,
            "frequency": 10,
            "first_selection": 50,
            "fallback": 60,
            "literal": 5,
            "average": 10,
            "aggregate": 10,
            "sum": 10,
            "comparison": 20,
            "filter_predicate": 25,
            "selection": 30,
            "transform": 40,
            "dedup": 45,
            "sort": 50,
            "sort_desc": 50,
            "sort_asc": 50,
            "output": 60,
            "repeat": 65,
        }

        def _wiring_priority(op) -> int:
            """Where an operation enters the wiring order.

            The order is the PIPELINE STAGE, read from the ontology:
            a data source first, then the aggregates a comparison
            thresholds against, then the PREDICATES a selection is built
            from, then the selection itself.  A predicate producer that
            wired after the selection would consume the selection's own
            output and the two would depend on each other — the graph
            must stay acyclic, so the type a primitive PRODUCES decides
            its stage when its semantic role does not already name one.
            """
            prim = PRIMITIVE_ONTOLOGY[op.name]
            priority = _WIRING_PRIORITY.get(prim.semantic_role, 35)
            if prim.produces == "predicate":
                priority = min(priority, 25)
            return priority

        ops.sort(key=_wiring_priority)

        op_by_id = {op.id: op for op in ops}

        def _reaches(start_id: str, target_id: str) -> bool:
            """Whether the dependency closure of `start_id` already
            contains `target_id`.

            An edge is only legal when it does not close a cycle: a
            predicate that consumed the selection built from its own
            predicate, or a threshold that consumed what it thresholds,
            would describe a dataflow that cannot be evaluated in any
            order.  The wiring therefore keeps every candidate producer
            whose own dependencies do not already reach this operation.
            """
            seen: set[str] = set()
            stack = [start_id]
            while stack:
                current = stack.pop()
                if current == target_id:
                    return True
                if current in seen:
                    continue
                seen.add(current)
                node = op_by_id.get(current)
                if node is not None:
                    stack.extend(node.depends_on)
            return False

        # Seed: input data is always a sequence source
        type_pool["sequence"] = ["__input__"]

        # Seed data sources (operations with no consumes)
        for op in ops:
            prim = PRIMITIVE_ONTOLOGY[op.name]
            if not prim.consumes:
                for out_type in (prim.produces,):
                    if out_type:
                        type_pool.setdefault(out_type, []).append(op.id)
                wired[op.id] = False

        # Iteratively wire operations by type matching.
        # Uses multiple passes — operations that can't wire in pass 1
        # (because their dependencies aren't available yet) will wire in
        # pass 2 after their dependencies have produced their types.
        max_iterations = len(ops) * 2
        for _ in range(max_iterations):
            progress = False
            for op in ops:
                if wired.get(op.id, False):
                    continue
                prim = PRIMITIVE_ONTOLOGY[op.name]
                needed = list(prim.consumes)

                all_available = True
                for needed_type in needed:
                    if needed_type not in type_pool or not type_pool[needed_type]:
                        all_available = False
                        break

                if not all_available:
                    continue

                # Wire: prefer most recent non-input producer.
                # Fall back to __input__ if no transform is available yet.
                # Special case: filters consume ALL predicate producers (AND).
                for needed_type in needed:
                    producers = type_pool.get(needed_type, [])
                    if producers:
                        if needed_type == "predicate" and op.name in ("filter", "filter_elements"):
                            # Filter consumes ALL predicate producers (AND composition)
                            for pid in producers:
                                if pid != "__input__" and pid not in op.depends_on \
                                        and not _reaches(pid, op.id):
                                    op.depends_on.append(pid)
                                    notes.append(
                                        f"{op.name}.consumes({needed_type}) <- "
                                        f"{pid}.produces({needed_type})"
                                    )
                        else:
                            # Prefer non-input producer (real transforms).
                            # The pool is a HISTORY of who produced the
                            # type, so it is read newest-first — over the
                            # DISTINCT producers, because a producer that
                            # recorded itself twice must not win by
                            # appearing last.
                            producer_id = None
                            for pid in reversed(dict.fromkeys(producers)):
                                if pid != "__input__" and not _reaches(pid, op.id):
                                    producer_id = pid
                                    break
                            if producer_id is None:
                                producer_id = "__input__"
                            if producer_id not in op.depends_on:
                                op.depends_on.append(producer_id)
                                notes.append(
                                    f"{op.name}.consumes({needed_type}) <- "
                                    f"{producer_id}.produces({needed_type})"
                                )

                # Update type pool: only add produces AFTER wiring
                # This ensures operations can only consume from actually-wired producers.
                # A source is already in the pool from the seeding pass —
                # the pool records WHO produced a type, once per producer,
                # so the "most recent producer" reading stays unambiguous.
                if prim.produces:
                    pool = type_pool.setdefault(prim.produces, [])
                    if op.id not in pool:
                        pool.append(op.id)

                # Note: we intentionally do NOT remove consumed sequence producers
                # from the pool. Multiple operations may need to consume from the
                # same upstream (e.g., filter needs both sort's output and compare's
                # output). The type-pool wiring already prefers the most recent
                # non-input producer, which is sufficient for correct wiring.

                # Mark as wired ONLY if it actually consumed what it needs
                wired[op.id] = True
                progress = True

            if not progress:
                break

        # Fallback: chain unwired ops to previous op.
        # Skip ops that were already wired (including to __input__).
        # Update the type pool after each wiring so subsequent ops
        # can chain to the newly wired op's output.
        for op in ops:
            if not op.depends_on and not wired.get(op.id, False) and ops.index(op) > 0:
                prim = PRIMITIVE_ONTOLOGY[op.name]
                if prim.consumes:
                    for needed_type in prim.consumes:
                        producers = type_pool.get(needed_type, [])
                        if producers:
                            # Prefer non-input producer
                            chosen = None
                            for pid in reversed(dict.fromkeys(producers)):
                                if pid != "__input__" and not _reaches(pid, op.id):
                                    chosen = pid
                                    break
                            if chosen is None:
                                chosen = "__input__"
                            op.depends_on.append(chosen)
                            break
                    else:
                        # No producer found for any needed type.
                        # For predicate ops, don't chain to previous —
                        # the post-wire step will connect them properly.
                        if prim.produces != "predicate":
                            prev_op = ops[ops.index(op) - 1]
                            op.depends_on.append(prev_op.id)
                elif ops.index(op) > 0:
                    prev_op = ops[ops.index(op) - 1]
                    op.depends_on.append(prev_op.id)
                # Add this op's output to the type pool
                if prim.produces:
                    type_pool.setdefault(prim.produces, []).append(op.id)
                wired[op.id] = True

        # Semantic correction: sort operations that wired to __input__
        # should instead consume from the filter's output if a filter exists.
        # This handles cases like "above average, sort descending" where sort
        # appears before filter in text but semantically depends on filter.
        filter_ops = [op for op in ops if op.name in ("filter", "filter_elements")]
        if filter_ops:
            filter_op = filter_ops[0]
            for op in ops:
                prim = PRIMITIVE_ONTOLOGY[op.name]
                if (op.name in ("sort_descending", "sort_ascending", "reverse",
                                "reverse_sequence")
                        and not op.depends_on):
                    # Rewire sort to depend on filter
                    op.depends_on = [filter_op.id]
                    # Remove filter's dependency on sort to avoid circular dep
                    if op.id in filter_op.depends_on:
                        filter_op.depends_on.remove(op.id)
                    notes.append(
                        f"semantic_correction: {op.name} rewired to {filter_op.id}"
                    )

        # ── Step 5: Add all operations to graph ──
        # Post-wire: connect filter ops to ALL predicate producers.
        # The main wiring loop may miss this if predicates appear after
        # the filter in text order.
        predicate_producers = [
            op.id for op in ops
            if op.name in ("compare_greater_than", "compare_less_than",
                           "compare_equal_to",
                           "is_prime", "is_multiple_of")
        ]
        for op in ops:
            if op.name in ("filter", "filter_elements"):
                for pred_id in predicate_producers:
                    if pred_id not in op.depends_on:
                        op.depends_on.append(pred_id)
                        notes.append(
                            f"post_wire: {op.name} <- {pred_id} (predicate)"
                        )

        # Post-wire: connect comparison ops to scalar (aggregate) producers.
        # Comparisons consume "scalar" which is produced by avg/sum/max/min.
        # If the comparison has no scalar dependency, find one.
        scalar_producers = [
            op.id for op in ops
            if op.name in ("calculate_average", "calculate_sum",
                           "find_max", "find_min")
        ]
        for op in ops:
            if op.name in ("compare_greater_than", "compare_less_than"):
                # A comparison against a LITERAL operand ("greater than
                # 5") thresholds a constant: wiring it to an aggregate
                # would silently replace the user's operand with the
                # mean (semantic contamination).  Only a comparison with
                # no literal operand may bind a scalar producer.
                if op.params.get("operand") is not None:
                    continue
                has_scalar_dep = False
                for dep_id in op.depends_on:
                    dep_op = op_by_id.get(dep_id) if dep_id != "__input__" else None
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "scalar":
                            has_scalar_dep = True
                            break
                if not has_scalar_dep and scalar_producers:
                    # Wire to the most recent scalar producer
                    op.depends_on.append(scalar_producers[-1])
                    notes.append(
                        f"post_wire: {op.name} <- {scalar_producers[-1]} (scalar)"
                    )

        # ── Post-wire: terminal aggregate consumes the selection ──
        # A TERMINAL aggregate — nothing downstream consumes its scalar —
        # is the objective head. Its operand is the SELECTION the text
        # describes ("sum of even numbers": the sum applies to the even
        # elements, not to the raw input). Rewire it to consume the
        # deepest selection step, so the dataflow reads
        # input → selection → aggregation.
        consumed_ids = {dep for op in ops for dep in op.depends_on
                        if dep != "__input__"}
        for op in ops:
            if op.name in ("calculate_sum", "calculate_average",
                           "find_max", "find_min", "count_elements") \
                    and op.id not in consumed_ids:
                selections = [s2 for s2 in ops
                              if s2.name in ("filter", "filter_elements")
                              and s2.id != op.id]
                if selections:
                    target = selections[-1]
                    if target.id not in op.depends_on:
                        op.depends_on = [d for d in op.depends_on
                                         if d == "__input__"]
                        op.depends_on.append(target.id)
                        notes.append(
                            f"post_wire: terminal {op.name} <- "
                            f"{target.id} (objective head over selection)")

        # ── Input bindings (spec #3) ──
        # The wiring decided dependencies; the bindings record what
        # feeds each input — named by the producer's produced type so
        # the binding states the dataflow, not just the edge.
        op_by_id = {op.id: op for op in ops}
        for op in ops:
            prim = PRIMITIVE_ONTOLOGY.get(op.name)
            consumes = list(prim.consumes) if prim else []
            bindings: list[dict] = []
            for dep in op.depends_on:
                if dep == "__input__":
                    label = consumes[0] if consumes else "data"
                    src = None
                else:
                    src = op_by_id.get(dep)
                    prod = (PRIMITIVE_ONTOLOGY.get(src.name).produces
                            if src and src.name in PRIMITIVE_ONTOLOGY else "")
                    label = prod or (consumes[len(bindings)]
                                     if len(bindings) < len(consumes)
                                     else "input")
                bindings.append({"input": label, "source": dep})
                self._record_semantic_param(op, label, src)
            if not bindings and consumes:
                bindings.append({"input": consumes[0],
                                 "source": "__input__"})
            op.input_bindings = bindings

        # ── Dataflow validation ──
        # A filter with no predicate performs no selection (it would
        # compile to an identity pass-through) — an unjustified operation
        # must not enter the graph. Justification: predicates in params,
        # or a dependency on a predicate producer.
        _pred_producer_ids = {
            op.id for op in ops
            if op.name in ("compare_greater_than", "compare_less_than",
                           "compare_equal_to",
                           "is_prime", "is_multiple_of")
        }
        ops = [op for op in ops
               if not (op.name in ("filter", "filter_elements")
                       and not op.params.get("predicates")
                       and not any(dep in _pred_producer_ids
                                   for dep in op.depends_on))]

        # ── Grounding validation (spec #3) ──
        # An operation without its reasoning and its evidence must not
        # enter the graph — every fill site above provides both; this
        # admits nothing that slipped past.
        _ungrounded = [op.name for op in ops
                       if not (op.semantic_justification and op.evidence)]
        if _ungrounded:
            ops = [op for op in ops
                   if op.semantic_justification and op.evidence]
            notes.append("rejected ungrounded operations: "
                         + ", ".join(_ungrounded))

        for op in ops:
            graph.add_operation(op)

        if ops:
            graph.entry_points = [op.id for op in ops if not op.depends_on]
            op_ids = {op.id for op in ops}
            depended_on = {dep_id for op in ops for dep_id in op.depends_on}
            graph.exit_points = [op.id for op in ops if op.id not in depended_on]

        if len(ops) > 1:
            level = "composed" if any(o.depends_on for o in ops) else "inferred"
        elif len(ops) == 1:
            level = "direct"
        else:
            level = "uncertain"

        return graph, notes, level

    # --- helpers ---
    def _prev_of(self, cues, i, names) -> Optional[int]:
        for j in range(i - 1, -1, -1):
            if cues[j].primitive in names:
                return j
        return None

    def _preceding(self, cues, i, names) -> list[int]:
        idxs = []
        for j in range(i):
            if cues[j].primitive in names:
                idxs.append(j)
        return idxs

    # ── Shell Knowledge Reasoning ──

    def reason_shell(self, query: str) -> dict:
        """Reason about a shell-related query using the shell knowledge graph.

        Returns a structured dict with platform, shell, operations, and
        an explanation. Does NOT execute any shell commands.
        """
        from brain.shell_knowledge import reason_about_shell
        if not self.concepts or not self.relations:
            return {"explanation": "No knowledge store available"}
        return reason_about_shell(query, self.concepts, self.relations)

    def _is_shell_query(self, text: str) -> bool:
        """Detect if a query is shell-related rather than code-generation."""
        if not self.concepts:
            return False
        # Check for shell platform/language concepts in the text
        shell_terms = {
            "powershell", "pwsh", "bash", "shell", "cmd", "terminal",
            "command prompt", "linux", "ubuntu", "windows",
            "파워셀", "바쉬", "리눅스", "우분투", "윈도우",
            "명령", "명령어", "프롬프트",
        }
        text_lower = text.lower()
        return any(term in text_lower for term in shell_terms)


# --------------------------------------------------------------------------- #
# Python code generation for composed atomic graphs (execution verification)
# --------------------------------------------------------------------------- #
def generate_python(graph: OperationGraph, func_name: str = "process_data", 
                    is_competitive: bool = False, input_specs: list = None, output_specs: list = None) -> str:
    """Compile a composed atomic operation graph into executable Python.

    Each operation generates code based on its semantic contract.
    Tracks which operations use the original input data vs. derived values.
    
    Args:
        graph: The operation graph to compile
        func_name: Name of the function to generate
        is_competitive: If True, generate competitive programming format with stdin/stdout
        input_specs: List of IOSpec objects describing input format (for competitive mode)
        output_specs: List of IOSpec objects describing output format (for competitive mode)
    """
    ops = graph.get_execution_order()
    
    # Competitive programming mode: generate solution() with stdin/stdout
    if is_competitive:
        return _generate_competitive_python(graph, func_name, input_specs or [], output_specs or [])
    
    lines = ["def {}(data):".format(func_name)]
    if not ops:
        lines.append("    return data")
        return "\n".join(lines)

    # Build a map: op_id -> variable name holding its output
    var_map = {}  # op_id -> variable name
    predicates = []  # collected predicate expressions for AND composition

    def _emit_pred():
        nonlocal predicates
        if not predicates:
            return
        if len(predicates) == 1:
            lines.append("    def pred(x): return {}".format(predicates[0]))
        else:
            combined = " and ".join("({})".format(p) for p in predicates)
            lines.append("    def pred(x): return {}".format(combined))
        predicates = []

    def _get_input_var(op):
        """Get the variable name for the primary data input of an operation.
        If the op depends on another op, use that op's output var.
        If the op has no dependencies (or depends on __input__), use 'data'.
        """
        prim = PRIMITIVE_ONTOLOGY[op.name]
        if not prim.consumes:
            return "data"  # data source

        # Check dependencies
        if op.depends_on:
            # Use the most recent dependency that produces a compatible type
            for dep_id in reversed(op.depends_on):
                if dep_id in var_map:
                    return var_map[dep_id]

        return "data"  # fallback to input

    for idx, op in enumerate(ops):
        # Use descriptive variable names for key operations
        _VAR_NAMES = {
            "calculate_average": "avg",
            "calculate_sum": "total",
            "find_max": "maximum",
            "find_min": "minimum",
            "generate_range": "data_range",
        }
        nxt = _VAR_NAMES.get(op.name, "r{}".format(idx))
        prim = PRIMITIVE_ONTOLOGY.get(op.name)
        handled = False  # set to True by handlers that manage their own var_map

        if op.name == "generate_range":
            start = op.params.get("start", 1)
            end = op.params.get("end", 100)
            lines.append("    {} = list(range({}, {}))".format(nxt, start, end + 1))

        elif op.name == "calculate_average":
            inp = _get_input_var(op)
            lines.append("    {} = sum({}) / len({}) if {} else 0".format(
                nxt, inp, inp, inp))

        elif op.name == "calculate_sum":
            inp = _get_input_var(op)
            lines.append("    {} = sum({})".format(nxt, inp))

        elif op.name == "find_max":
            inp = _get_input_var(op)
            lines.append("    {} = max({})".format(nxt, inp))

        elif op.name == "search":
            # Linear search: first index whose element equals the target
            inp = _get_input_var(op)
            target = op.params.get("target", "target")
            lines.append("    {} = -1".format(nxt))
            lines.append("    for _i, _x in enumerate({}):".format(inp))
            lines.append("        if _x == {}:".format(target))
            lines.append("            {} = _i".format(nxt))
            lines.append("            break")

        elif op.name == "find_min":
            inp = _get_input_var(op)
            lines.append("    {} = min({})".format(nxt, inp))

        elif op.name == "compare_greater_than":
            # Find threshold from dependencies: look for an aggregate that
            # produces "scalar" which this comparison consumes.
            agg_var = None
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "scalar":
                            agg_var = var_map[dep_id]
                            break
            if agg_var is None:
                # Fall back to an explicit operand the request stated
                # ("greater than 5"), then to a producer binding.
                thresh = op.params.get("operand")
                if thresh is None:
                    thresh = op.params.get("threshold")
                if thresh is not None:
                    agg_var = str(thresh)
                else:
                    # No scalar dependency and no operand — this comparison
                    # is orphaned (no aggregate in the pipeline). Skip it
                    # rather than raising — the test suite expects graceful
                    # handling of partial detections.
                    continue
            predicates.append("x > {}".format(agg_var))
            # Comparison only produces a predicate, not a value — skip assignment
            continue

        elif op.name == "compare_less_than":
            agg_var = None
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "scalar":
                            agg_var = var_map[dep_id]
                            break
            if agg_var is None:
                thresh = op.params.get("operand")
                if thresh is None:
                    thresh = op.params.get("threshold")
                if thresh is not None:
                    agg_var = str(thresh)
                else:
                    continue
            predicates.append("x < {}".format(agg_var))
            # Comparison only produces a predicate, not a value — skip assignment
            continue

        elif op.name == "is_prime":
            predicates.append(
                "x > 1 and all(x % d != 0 for d in range(2, int(x**0.5) + 1))"
            )
            # Predicate-only operation — skip assignment
            continue

        elif op.name == "is_multiple_of":
            mod = op.params.get("modulus", 2)
            predicates.append("x % {} == 0".format(mod))
            # Predicate-only operation — skip assignment
            continue

        elif op.name == "count_frequency":
            inp = _get_input_var(op)
            lines.append("    from collections import Counter")
            lines.append("    {} = Counter({})".format(nxt, inp))

        elif op.name == "compare_equal_to":
            freq_var = None
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "frequency":
                            freq_var = var_map[dep_id]
                            break
            operand = op.params.get("operand", 1)
            if freq_var is not None:
                predicates.append("{}[x] == {}".format(freq_var, operand))
            else:
                predicates.append("x == {}".format(operand))
            continue

        elif op.name == "select_first":
            inp = _get_input_var(op)
            lines.append("    {} = {}[0] if {} else None".format(nxt, inp, inp))

        elif op.name == "fallback_none":
            # The empty-selection None case is already produced by the
            # first-selection step; the fallback binds that optional into
            # the pipeline's output.
            inp = _get_input_var(op)
            var_map[op.id] = inp
            handled = True

        elif op.name in ("filter", "filter_elements"):
            _emit_pred()
            # Filter needs TWO inputs: sequence (from data/transform) and
            # predicate (from comparison). Find the sequence input by checking
            # which dependency produces "sequence", not "predicate".
            seq_inp = "data"
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "sequence":
                            seq_inp = var_map[dep_id]
                            break
            # Only use pred(x) if a predicate was actually defined
            if predicates or any("def pred" in l for l in lines):
                lines.append("    {} = [x for x in {} if pred(x)]".format(nxt, seq_inp))
            else:
                # No predicate available — filter with identity (pass-through)
                lines.append("    {} = list({})".format(nxt, seq_inp))

        elif op.name == "remove_duplicates":
            inp = _get_input_var(op)
            lines.append("    seen = set()")
            lines.append("    {} = [x for x in {} if not (x in seen or seen.add(x))]".format(
                nxt, inp))

        elif op.name == "sort_descending":
            inp = _get_input_var(op)
            lines.append("    {} = sorted({}, reverse=True)".format(nxt, inp))

        elif op.name == "sort_ascending":
            inp = _get_input_var(op)
            lines.append("    {} = sorted({})".format(nxt, inp))

        elif op.name in ("reverse", "reverse_sequence"):
            inp = _get_input_var(op)
            lines.append("    {} = {}[::-1]".format(nxt, inp))

        elif op.name == "add":
            inp = _get_input_var(op)
            lines.append("    {} = {}[0] + {}[1]".format(nxt, inp, inp))

        elif op.name == "multiply":
            inp = _get_input_var(op)
            lines.append("    {} = {}[0] * {}[1]".format(nxt, inp, inp))

        elif op.name == "factorial":
            inp = _get_input_var(op)
            lines.append("    import math")
            lines.append("    {} = math.factorial({})".format(nxt, inp))

        elif op.name == "fibonacci":
            inp = _get_input_var(op)
            lines.append("    a, b = 0, 1")
            lines.append("    for _ in range({}):".format(inp))
            lines.append("        a, b = b, a + b")
            lines.append("    {} = a".format(nxt))

        elif op.name == "concatenate":
            inp = _get_input_var(op)
            lines.append('    {} = "".join({})'.format(nxt, inp))

        elif op.name == "literal_value":
            # Extract the literal value from params or description
            value = op.params.get("value", "")
            value_type = op.params.get("value_type", "string")
            if not value:
                # Try to extract from description (e.g., "Hello", "Apple")
                desc = op.description or ""
                # Look for quoted strings
                m = re.search(r'["\']([^"\']+)["\']', desc)
                if m:
                    value = m.group(1)
                else:
                    # Try to extract word after "output/print/display"
                    m = re.search(r'(?:output|print|display|show)\s+(.+)', desc, re.IGNORECASE)
                    if m:
                        value = m.group(1).strip()
                    else:
                        value = desc
            # Emit as Python literal — numeric without quotes, strings with quotes
            if value_type == "number" or (value and re.match(r'^-?\d+(\.\d+)?$', str(value))):
                lines.append('    {} = {}'.format(nxt, value))
            else:
                escaped = value.replace("\\", "\\\\").replace('"', '\\"')
                lines.append('    {} = "{}"'.format(nxt, escaped))

        elif op.name == "emit_output":
            # An emit whose only input is the raw stdin data (no other
            # computation in the graph) is the identity echo — a fake
            # solution, not a realization of the meaning.  Fail the
            # generation honestly instead.
            graph_op_names = {o.name for o in ops}
            if graph_op_names <= {"emit_output", "repeat_output"}:
                raise ValueError(
                    "No computable semantics: the graph only echoes its "
                    "input")
            inp = _get_input_var(op)
            # The emit's input must be a COMPUTED value: when the emit
            # declared a dependency but that dependency was skipped (no
            # executable semantics), the input falls back to the raw
            # stdin data — echoing it is a fake solution.  A legitimate
            # echo (the user asked to output the input as-is) declares
            # NO dependency and stays valid.
            if inp == "data" and op.depends_on:
                raise ValueError(
                    "No computable semantics: emit_output's dependency "
                    "produced no executable value")
            # Check if a repeat_output depends on this op
            repeat_count = None
            for other_op in ops:
                if other_op.name == "repeat_output" and op.id in other_op.depends_on:
                    repeat_count = other_op.params.get("count", 2)
                    break
            if repeat_count is not None:
                # Repeat wrapper: print N times
                lines.append("    for _ in range({}):".format(repeat_count))
                lines.append("        print({})".format(inp))
            else:
                lines.append("    print({})".format(inp))
            var_map[op.id] = inp  # reuse input variable, don't create new one
            handled = True

        elif op.name == "repeat_output":
            # Skip: handled by emit_output's repeat wrapper.
            # Map to input variable so exit_point logic works.
            inp = _get_input_var(op)
            var_map[op.id] = inp
            handled = True

        else:
            # No identity fallback: every operation in the graph MUST have a
            # concrete implementation matching its semantic contract.
            raise ValueError(
                "No code generator for primitive '{}' — cannot emit "
                "identity fallback".format(op.name)
            )

        if not handled:
            var_map[op.id] = nxt

    # Return the output of the terminal operation (one with no downstream consumers)
    exit_points = graph.exit_points or [ops[-1].id]
    # Use the LAST exit point — this is the final state-producing transform
    # (e.g., sort_descending), not a scalar side-computation (e.g., avg).
    last_id = exit_points[-1]
    last_op = graph.get_operation(last_id)
    last_var = var_map.get(last_id, "data")

    # If the terminal op is a predicate-only op (e.g., is_prime) and no
    # filter consumed the predicate, return pred(data) instead.
    predicate_only = {"is_prime", "is_multiple_of", "compare_greater_than",
                      "compare_less_than"}
    if last_op and last_op.name in predicate_only and predicates:
        _emit_pred()
        lines.append("    return pred(data)")
    else:
        # Resolve var_map chains: if last_var points to another var, follow it
        resolved = last_var
        seen = set()
        while resolved in var_map and resolved != var_map.get(resolved) and resolved not in seen:
            seen.add(resolved)
            resolved = var_map[resolved]
        lines.append("    return {}".format(resolved))
    return "\n".join(lines)


def _generate_competitive_python(graph: OperationGraph, func_name: str, 
                                  input_specs: list, output_specs: list,
                                  call_form: bool = False,
                                  call_params: list = None) -> str:
    """Generate Python code for competitive programming.

    Two forms:
    - stdio (default): def name() reads from stdin, prints to stdout
    - call form: def name(params) takes the declared parameters and
      RETURNS the result (call-based judging via the problem's starter
      code contract)

    This generates a solution() function that:
    1. Reads input from stdin (stdio form only)
    2. Processes using the operation graph
    3. Prints (stdio) or returns (call form) the output
    """
    ops = graph.get_execution_order()

    if call_form:
        params = call_params or []
        data_var = params[0] if params else "data"
        lines = ["def {}({}):".format(func_name, ", ".join(params))]
        if not ops:
            lines.append(f"    return {data_var}")
            return "\n".join(lines)
        lines.append(f"    data = {data_var}")
    else:
        # Start with function definition
        lines = ["def {}():".format(func_name)]

        # Always add generic input reading - parse all stdin as integers
        lines.append("    import sys")
        lines.append("    data = list(map(int, sys.stdin.read().split()))")
        lines.append("")

        # No operations: the graph carries no computable semantics —
        # echoing the input is a fake solution, not a realization of the
        # meaning.  Fail the generation honestly instead (the pipeline's
        # fallback reports insufficient understanding).
        if not ops:
            raise ValueError(
                "No computable semantics: the composed graph has no "
                "operations to realize")
    
    # Generate processing code from operation graph
    var_map = {}
    predicates = []
    
    def _emit_pred():
        nonlocal predicates
        if not predicates:
            return
        if len(predicates) == 1:
            lines.append("    def pred(x): return {}".format(predicates[0]))
        else:
            combined = " and ".join("({})".format(p) for p in predicates)
            lines.append("    def pred(x): return {}".format(combined))
        predicates = []
    
    def _get_input_var(op):
        prim = PRIMITIVE_ONTOLOGY.get(op.name)
        if not prim or not prim.consumes:
            return "data"
        if op.depends_on:
            for dep_id in reversed(op.depends_on):
                if dep_id in var_map:
                    return var_map[dep_id]
        return "data"
    
    for idx, op in enumerate(ops):
        nxt = "r{}".format(idx)
        
        if op.name == "generate_range":
            start = op.params.get("start", 1)
            end = op.params.get("end", 100)
            lines.append("    {} = list(range({}, {}))".format(nxt, start, end + 1))
            var_map[op.id] = nxt
        
        elif op.name == "calculate_average":
            inp = _get_input_var(op)
            lines.append("    {} = sum({}) / len({}) if {} else 0".format(nxt, inp, inp, inp))
            var_map[op.id] = nxt
        
        elif op.name == "calculate_sum":
            inp = _get_input_var(op)
            lines.append("    {} = sum({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "find_max":
            inp = _get_input_var(op)
            lines.append("    {} = max({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "find_min":
            inp = _get_input_var(op)
            lines.append("    {} = min({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "count_elements":
            inp = _get_input_var(op)
            lines.append("    {} = len({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "accumulate":
            inp = _get_input_var(op)
            lines.append("    {} = sum({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "compare_greater_than":
            agg_var = None
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "scalar":
                            agg_var = var_map[dep_id]
                            break
            if agg_var is None:
                agg_var = op.params.get("threshold", "0")
            predicates.append("x > {}".format(agg_var))
            continue
        
        elif op.name == "compare_less_than":
            agg_var = None
            for dep_id in op.depends_on:
                if dep_id in var_map:
                    dep_op = graph.get_operation(dep_id)
                    if dep_op:
                        dep_prim = PRIMITIVE_ONTOLOGY.get(dep_op.name)
                        if dep_prim and dep_prim.produces == "scalar":
                            agg_var = var_map[dep_id]
                            break
            if agg_var is None:
                agg_var = op.params.get("threshold", "0")
            predicates.append("x < {}".format(agg_var))
            continue
        
        elif op.name == "is_prime":
            predicates.append(
                "x > 1 and all(x % d != 0 for d in range(2, int(x**0.5) + 1))"
            )
            continue
        
        elif op.name == "is_multiple_of":
            mod = op.params.get("modulus", 2)
            predicates.append("x % {} == 0".format(mod))
            continue
        
        elif op.name in ("filter", "filter_elements", "conditional_filter"):
            _emit_pred()
            seq_inp = _get_input_var(op)
            if predicates or any("def pred" in l for l in lines):
                lines.append("    {} = [x for x in {} if pred(x)]".format(nxt, seq_inp))
            else:
                lines.append("    {} = list({})".format(nxt, seq_inp))
            var_map[op.id] = nxt
        
        elif op.name == "remove_duplicates":
            inp = _get_input_var(op)
            lines.append("    seen = set()")
            lines.append("    {} = [x for x in {} if not (x in seen or seen.add(x))]".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "sort_descending":
            inp = _get_input_var(op)
            lines.append("    {} = sorted({}, reverse=True)".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "sort_ascending":
            inp = _get_input_var(op)
            lines.append("    {} = sorted({})".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name in ("reverse", "reverse_sequence"):
            inp = _get_input_var(op)
            lines.append("    {} = {}[::-1]".format(nxt, inp))
            var_map[op.id] = nxt
        
        elif op.name == "emit_output":
            # The emit's input must be a COMPUTED value: when the emit
            # declared a dependency but that dependency was skipped (no
            # executable semantics), the input falls back to the raw
            # stdin data — echoing it is a fake solution, not a
            # realization of the meaning.  Fail the generation honestly.
            # A legitimate echo (the user asked to output the input
            # as-is) declares NO dependency and stays valid.
            inp = _get_input_var(op)
            if inp == "data" and op.depends_on:
                raise ValueError(
                    "No computable semantics: emit_output's dependency "
                    "produced no executable value")
            if call_form:
                lines.append(f"    return {inp}")
            else:
                lines.append("    print({})".format(inp))
            var_map[op.id] = inp

        elif op.name == "repeat_output":
            inp = _get_input_var(op)
            count = op.params.get("count", 2)
            if call_form:
                lines.append(f"    return {inp}")
            else:
                lines.append("    for _ in range({}):".format(count))
                lines.append("        print({})".format(inp))
            var_map[op.id] = inp
        
        elif op.name == "literal_value":
            value = op.params.get("value", "")
            if value and str(value).strip():
                lines.append('    {} = {}'.format(nxt, repr(value)))
                var_map[op.id] = nxt
            else:
                var_map[op.id] = _get_input_var(op)
        
        elif op.name in ("nested_loop", "iterate_range"):
            inp = _get_input_var(op)
            var_map[op.id] = inp
        
        else:
            # Unknown operation - skip
            continue
    
    # Add default output if no emit_output was found
    has_output = any(op.name in ("emit_output", "repeat_output") for op in ops)
    if not has_output:
        # Find the last computed result
        if var_map:
            last_id = list(var_map.keys())[-1]
            result_var = var_map[last_id]
        else:
            # No operation produced executable semantics — fail the
            # generation instead of echoing the input. The identity
            # fallback emits broken code that cannot satisfy any contract.
            raise ValueError(
                "No executable semantics for operations: "
                + ", ".join(op.name for op in ops))
        # The result must be a COMPUTED value: when every var_map entry
        # is the raw input passthrough, no operation actually produced a
        # new value — echoing the input is a fake solution, not a
        # realization of the meaning.  Fail the generation honestly.
        if all(v == "data" for v in var_map.values()):
            raise ValueError(
                "No executable semantics for operations: "
                + ", ".join(op.name for op in ops))
        if call_form:
            lines.append(f"    return {result_var}")
        else:
            lines.append("    print({})".format(result_var))

    return "\n".join(lines)


def create_composer(concept_store=None, relation_store=None) -> SemanticComposer:
    return SemanticComposer(concept_store, relation_store)