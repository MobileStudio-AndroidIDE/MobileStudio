"""Pipeline contract: the element domain and objective the pipeline realizes.

The semantic layer declares WHAT the function takes, what it returns and
which objective it realizes.  Every generator — the IR builder, the
language renderers and the behavioral oracle — consumes ONE contract, so
the four descriptions of a pipeline can never diverge (spec #5):

    Semantic Output Type == IR Output Type
        == Generated Function Return Type == Verified Behavioral Type

The contract is derived from the semantic representation only; it is
never re-read from the request text by a later stage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# The element domain can only be one of these language-level types
# (an undeclared domain is "", which means "derive it from the
# operation vocabulary" — the pre-contract behavior).
_KNOWN_ELEMS = {"str", "int", "float", "bool"}


@dataclass
class PipelineContract:
    """What the pipeline consumes, produces and optimizes."""

    # Language-level element domain: "str" | "int" | "float" | "bool" | ""
    elem_type: str = ""
    # The measure an extremum selects by: "length" | "" (by value).
    # "length" means find_max/find_min compares the DERIVED length of
    # each element, not the element's own ordering.
    select_key: str = ""
    # The objective the pipeline realizes (carried into the IR so the
    # trace can show the meaning surviving generation).
    direction: str = ""
    measure: str = ""
    target: str = ""
    evidence: list[str] = field(default_factory=list)
    # A LITERAL operand the request supplied for a selection ("greater
    # than 5").  None means the threshold is derived (the aggregate's
    # average) — a selection with neither operand is not renderable and
    # must fail honestly instead of inventing a threshold.
    threshold: Optional[float] = None
    # Per-operation parameters from the validated plan (e.g. the
    # equality operand).  The behavioral oracle reads the SAME plan the
    # renderer read — one contract, never a re-derivation.
    op_params: dict = field(default_factory=dict)

    @property
    def keyed(self) -> bool:
        """Does the extremum select by a derived key rather than by value?"""
        return self.select_key == "length"

    def with_elem(self, elem_type: str) -> "PipelineContract":
        e = (elem_type or "").strip().lower()
        return PipelineContract(
            elem_type=e if e in _KNOWN_ELEMS else "",
            select_key=self.select_key,
            direction=self.direction,
            measure=self.measure,
            target=self.target,
            evidence=list(self.evidence),
            threshold=self.threshold,
            op_params=dict(self.op_params),
        )


def contract_from_semantic(semantic) -> PipelineContract:
    """Build the contract from the semantic representation.

    Element domain: the declared input element (List<String> → str),
    falling back to "" when the semantics declare none.  Select key: the
    objective's measure — length selection is a property of the
    objective, never of a word in the sentence.
    """
    if semantic is None:
        return PipelineContract()

    from brain.semantic_parser import SemanticParser

    element = SemanticParser._element_type(semantic)
    elem = SemanticParser._lang_type(element) if element else ""

    direction = measure = target = ""
    evidence: list[str] = []
    objectives = getattr(semantic, "objectives", None) or []
    if objectives:
        obj = objectives[0]
        direction = getattr(obj, "direction", "") or ""
        measure = getattr(obj, "measure", "") or ""
        target = getattr(obj, "target", "") or ""
        ev = getattr(obj, "evidence", "") or ""
        if ev:
            evidence.append(ev)

    select_key = "length" if measure == "length" else ""

    params_by_op: dict = {}
    composition = getattr(semantic, "composition", None) or {}
    _graph = composition.get("graph") if isinstance(composition, dict) else None
    for _op in ((_graph or {}).get("operations") or []):
        if isinstance(_op, dict) and _op.get("name") and _op.get("params"):
            params_by_op.setdefault(str(_op["name"]), dict(_op["params"]))
    _og = getattr(semantic, "operation_graph", None)
    for _op in (getattr(_og, "operations", None) or []):
        _name = getattr(_op, "name", "")
        _params = getattr(_op, "params", None)
        if _name and _params:
            params_by_op.setdefault(str(_name), dict(_params))

    return PipelineContract(
        elem_type=elem if elem in _KNOWN_ELEMS else "",
        select_key=select_key,
        direction=direction,
        measure=measure,
        target=target,
        evidence=evidence,
        threshold=_literal_threshold(semantic),
        op_params=params_by_op,
    )


def _literal_threshold(semantic) -> Optional[float]:
    """The literal selection operand the semantics recorded, if any.

    The composer stores it on the comparison op's params as "operand"
    (a numeric threshold).  A named producer in the same slot ("avg")
    is a DATAFLOW binding, never a threshold — only real numbers count.
    """
    def _scan(params):
        if not isinstance(params, dict):
            return None
        for key in ("operand", "threshold_value"):
            val = params.get(key)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return float(val)
        return None

    composition = getattr(semantic, "composition", None) or {}
    graph = composition.get("graph") if isinstance(composition, dict) else None
    ops = (graph or {}).get("operations") if isinstance(graph, dict) else None
    for op in (ops or []):
        if isinstance(op, dict):
            val = _scan(op.get("params"))
            if val is not None:
                return val
    og = getattr(semantic, "operation_graph", None)
    for op in (getattr(og, "operations", None) or []):
        val = _scan(getattr(op, "params", None))
        if val is not None:
            return val
    return None
