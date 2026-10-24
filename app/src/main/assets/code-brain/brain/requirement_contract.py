"""User Requirement Contract — the authority behavioral verification answers to.

The pipeline is:

    User Semantics -> User Requirement Contract -> Semantic Representation
        -> Operation Graph -> IR -> Code -> Runtime
        -> Behavioral Verification -> Requirement Coverage

The OPERATION CONTRACT (the normalized op list the generator renders) is an
IMPLEMENTATION-internal artifact: it only says what the generated code was
supposed to compute.  The final correctness authority is the USER
REQUIREMENT CONTRACT — the requirements extracted from what the user asked,
each with its own evidence in the request text.

Every requirement item carries its coverage across the stages:

    semantic -> plan -> IR -> code -> runtime -> behavioral_test -> verified

A generation PASSES only when:

  * every required item is verified,
  * the semantic / IR / code / runtime output types agree,
  * behavior was verified against the requirement oracle (never a smoke run),
  * no operation lacks a requirement justification,
  * no unsupported fallback was taken.

This module is deliberately free of sentence keyword tables: items are the
validated operation plan plus the text-level CONSTRAINTS the semantics
recorded (e.g. "preserving the original order" forbids reordering).  The
constraint's evidence is the phrase that grounded it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Honest stage status for work that was never executed.  Existing flows use
# booleans (True/False) and a missing key means "not reached"; NOT_RUN states
# explicitly "this stage did not run", so it can never be mistaken for pass.
NOT_RUN = "not_run"


def mark_not_run(stages: dict, stage: str, reason: str) -> dict:
    """Record an honest NOT_RUN for a coverage stage, with its reason.

    The reason is stored under ``stages["not_run_reasons"][stage]`` so the
    status value itself stays a single comparable token.
    """
    stages[stage] = NOT_RUN
    reasons = stages.get("not_run_reasons")
    if not isinstance(reasons, dict):
        reasons = {}
        stages["not_run_reasons"] = reasons
    reasons[stage] = str(reason)
    return stages


_AGGREGATE_OPS = {
    "calculate_average", "calculate_sum", "find_max", "find_min",
    "count_elements", "add", "multiply",
}
_SELECTION_OPS = {
    "filter", "filter_above", "filter_below", "filter_even",
}
_ORDERING_OPS = {
    "sort_ascending", "sort_descending", "reverse", "reverse_sequence",
}
_TRANSFORM_OPS = {
    "remove_duplicates", "concatenate",
}
_SCALAR_OUTPUT_OPS = _AGGREGATE_OPS | {"search", "fibonacci", "factorial"}


def operation_kind(op: str) -> str:
    """The requirement kind an operation realizes."""
    if op in _AGGREGATE_OPS:
        return "aggregation"
    if op in _SELECTION_OPS:
        return "selection"
    if op in _ORDERING_OPS:
        return "ordering"
    if op in _TRANSFORM_OPS:
        return "transform"
    return "transform"


def output_kind(type_name: str) -> str:
    """Classify a declared type into a language-level output kind.

    Returns "" for an undeclared type (unknown — never a mismatch), and
    "optional" for a nullable scalar ("char | None" → str or none).
    """
    t = (type_name or "").strip().lower()
    if not t:
        return ""
    if t.startswith("optional") or t.startswith("option<"):
        inner = t[len("optional"):] if t.startswith("optional") else t[len("option<"):]
        inner = inner.strip().rstrip(">").strip(" []").strip()
        if not inner or inner == "any":
            return "optional:any"
        inner_kind = output_kind(inner)
        return "optional:" + (inner_kind or "any")
    if "|" in t:
        inner = t.split("|")[0].strip()
        inner_kind = output_kind(inner)
        return "optional:" + (inner_kind or "any")
    if t.startswith("list") or t.startswith("vec<") or t.endswith("[]") \
            or t.startswith("array") or t.startswith("sequence") \
            or t.startswith("set") or t.startswith("[]"):
        return "list"
    if t in ("str", "string", "char", "character", "text") \
            or t.startswith("string"):
        return "str"
    if t in ("bool", "boolean"):
        return "bool"
    if t in ("none", "null", "nil"):
        return "none"
    if t in ("int", "integer", "float", "double", "number", "long",
             "i32", "i64", "f32", "f64", "decimal", "real"):
        return "number"
    return ""


def runtime_output_kind(code: str, func_name: str,
                        expected_kind: str = "") -> str:
    """Execute the generated function on structural probes and classify
    the RUNTIME output type.

    Probes are drawn from several domains (numbers, strings, empty);
    the outcome matching the EXPECTED kind wins when the semantics
    declared one — the probe domain must not fabricate a mismatch (a
    string-domain function probed with an int list would "return" a
    number).  "unavailable" is reported honestly when nothing runs.
    """
    if not code or not func_name:
        return "unavailable"
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception:
        return "unavailable"
    fn = namespace.get(func_name)
    if not callable(fn):
        return "unavailable"
    probes = (
        [3, 1, 3, 2, 1],
        [1.5, -2.0, 0.0, 1.5],
        ["apple", "pear", "banana"],
        "aabbc",
        "aabb",
        [5, 5, 5],
        [],
    )
    outcomes: list[str] = []
    for probe in probes:
        try:
            out = fn(probe)
        except Exception:
            continue
        outcomes.append(_kind_of_value(out))
    if not outcomes:
        return "unavailable"
    pref = expected_kind
    if pref.startswith("optional:"):
        pref = pref.split(":", 1)[1]
    if pref and pref in outcomes:
        return pref
    if pref != "any" and "none" in outcomes and expected_kind.startswith("optional"):
        return "none"
    return outcomes[0]


def _kind_of_value(value) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple, set, dict)):
        return "list"
    if value is None:
        return "none"
    return "other"


def kinds_match(expected: str, actual: str) -> Optional[bool]:
    """Whether a runtime kind satisfies an expected declared kind.

    None = not comparable (unknown expectation or unavailable probe).
    """
    if not expected or actual in ("unavailable", ""):
        return None
    if expected.startswith("optional:"):
        inner = expected.split(":", 1)[1]
        if inner == "any":
            return True
        return actual in ("none", inner)
    return expected == actual


@dataclass
class RequirementItem:
    """One requirement the user stated, with its evidence and coverage."""

    id: str
    kind: str                    # aggregation | selection | ordering | transform | constraint
    operation: str               # the pipeline op that realizes it ("" for a constraint)
    semantic: str                # canonical meaning
    evidence: str = ""           # the request span that grounds it
    required: bool = True
    stages: dict = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        value = self.stages.get("verified")
        return bool(value) and value != NOT_RUN

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "operation": self.operation,
            "semantic": self.semantic,
            "evidence": self.evidence,
            "required": self.required,
            "stages": dict(self.stages),
            "verified": self.verified,
        }


@dataclass
class RequirementContract:
    """The structured requirements a request states."""

    request: str = ""
    items: list[RequirementItem] = field(default_factory=list)
    output_type: str = ""
    input_type: str = ""
    threshold: Optional[float] = None

    def required_items(self) -> list[RequirementItem]:
        return [i for i in self.items if i.required]

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "output_type": self.output_type,
            "input_type": self.input_type,
            "items": [i.to_dict() for i in self.items],
        }


def _evidence_for(op_name: str, semantic) -> str:
    """The evidence the semantics recorded for an operation."""
    comp = getattr(semantic, "composition", None) or {}
    graph = comp.get("graph") if isinstance(comp, dict) else None
    for entry in ((graph or {}).get("operations") or []):
        if isinstance(entry, dict) and entry.get("name") == op_name:
            ev = entry.get("evidence") or []
            if ev:
                return "; ".join(str(e) for e in ev)
            sj = entry.get("semantic_justification") or ""
            if sj:
                return str(sj)
            break
    og = getattr(semantic, "operation_graph", None)
    for op in (getattr(og, "operations", None) or []):
        if getattr(op, "name", "") == op_name:
            ev = getattr(op, "evidence", None) or []
            if ev:
                return "; ".join(str(e) for e in ev)
            return str(getattr(op, "semantic_justification", "") or "")
    return ""


def _constraints(semantic) -> list[dict]:
    comp = getattr(semantic, "composition", None) or {}
    if isinstance(comp, dict):
        cons = comp.get("constraints")
        if isinstance(cons, list):
            return [c for c in cons if isinstance(c, dict)]
    return []


def _unjustified_ops(semantic, ops: list[str]) -> list[str]:
    """Operations with no requirement justification or evidence.

    The graph validation should already have removed them; this is the
    final audit the coverage matrix reports (never silently ignored).
    """
    unjustified: list[str] = []
    comp = getattr(semantic, "composition", None) or {}
    graph = comp.get("graph") if isinstance(comp, dict) else None
    entries = {
        e.get("name"): e for e in ((graph or {}).get("operations") or [])
        if isinstance(e, dict)
    }
    for op in ops:
        entry = entries.get(op)
        if entry is None:
            continue  # planner-derived names (normalized) have no entry
        if not (entry.get("semantic_justification") and entry.get("evidence")):
            unjustified.append(op)
    return unjustified


def build_requirement_contract(request: str, semantic, ops: list[str],
                               threshold: Optional[float] = None,
                               output_type: str = "") -> RequirementContract:
    """Build the requirement contract from the VALIDATED operation plan.

    The plan is the validated graph (unjustified operations already
    removed); each operation becomes a requirement item, and every
    text-level constraint the semantics recorded becomes its own
    requirement.  The contract's output type is the terminal operation's
    contract — the SAME derivation the IR and the renderers read.
    """
    ops = [str(o) for o in (ops or [])]
    declared_out = (output_type
                    or str(getattr(semantic, "output_type", "") or ""))
    contract = RequirementContract(
        request=request or "",
        output_type=declared_out,
        input_type=_input_type(semantic),
        threshold=threshold,
    )
    used: dict[str, int] = {}
    index = 0
    for op in ops:
        index += 1
        used[op] = used.get(op, 0) + 1
        evidence = _evidence_for(op, semantic)
        contract.items.append(RequirementItem(
            id=f"R{index}",
            kind=operation_kind(op),
            operation=op,
            semantic=(f"{op}({threshold})"
                      if threshold is not None and operation_kind(op) == "selection"
                      else op),
            evidence=evidence,
        ))
    for constraint in _constraints(semantic):
        index += 1
        contract.items.append(RequirementItem(
            id=f"R{index}",
            kind="constraint",
            operation="",
            semantic=str(constraint.get("kind") or "constraint"),
            evidence=str(constraint.get("evidence") or ""),
        ))
    return contract


def _input_type(semantic) -> str:
    inputs = getattr(semantic, "inputs", None) or []
    if inputs:
        return str(getattr(inputs[0], "type_signature", "") or "")
    params = getattr(getattr(semantic, "function_signature", None),
                     "parameters", None) or []
    return str(params[0]) if params else ""


def evaluate_contract(contract: RequirementContract, *,
                      ir_ops: list[str],
                      ir_type: str,
                      py_code: str,
                      runtime_kind: str,
                      behavioral_ok: bool,
                      unjustified_ops: Optional[list[str]] = None,
                      oracle_ran: bool = False) -> dict:
    """Evaluate the requirement coverage matrix and the final PASS.

    Both the IR operations and the generated code come from the SAME
    validated plan, so an item's code coverage is the presence of its
    operation in that plan, and runtime/behavioral coverage is the
    behavioral verification's result against the requirement oracle.
    """
    ir_ops = [str(o) for o in (ir_ops or [])]
    plan_ops = list(ir_ops)
    has_ir = bool(ir_ops)
    has_code = bool(py_code and py_code.strip())
    behavioral = bool(behavioral_ok and oracle_ran)
    reordering_present = any(o in _ORDERING_OPS for o in plan_ops)

    for item in contract.items:
        if item.kind == "constraint":
            constraint_ok = True
            if item.semantic == "preserve_order":
                # The result must keep the input's order: any ordering
                # operation in the validated plan contradicts it.
                constraint_ok = not reordering_present
            stages = {
                "semantic": True,
                "plan": True,
                "ir": has_ir and constraint_ok,
                "code": has_code and constraint_ok,
                "runtime": behavioral and constraint_ok,
                "behavioral_test": behavioral and constraint_ok,
            }
            stages["verified"] = all(stages.values())
            item.stages = stages
            continue

        in_plan = item.operation in plan_ops
        stages = {
            "semantic": True,
            "plan": in_plan,
            "ir": has_ir and in_plan,
            "code": has_code and in_plan,
            "runtime": behavioral and in_plan,
            "behavioral_test": behavioral and in_plan,
        }
        stages["verified"] = all(stages.values())
        item.stages = stages

    required = contract.required_items()
    verified = sum(1 for i in required if i.verified)
    total = len(required)

    expected_kind = output_kind(contract.output_type)
    match = kinds_match(expected_kind, runtime_kind)
    if match is None:
        type_consistency = "not_comparable"
    elif match:
        type_consistency = "pass"
    else:
        type_consistency = "fail"

    ir_kind = output_kind(ir_type)
    ir_matches = True
    if expected_kind and ir_kind and expected_kind != ir_kind:
        ir_matches = False

    unjustified = list(unjustified_ops or [])
    passed = bool(
        total > 0
        and verified == total
        and behavioral
        and type_consistency != "fail"
        and ir_matches
        and not unjustified
    )

    return {
        "passed": passed,
        "verified": verified,
        "total": total,
        "verified_ratio": round(verified / total, 3) if total else 0.0,
        "behavioral": behavioral,
        "oracle_ran": bool(oracle_ran),
        "type_consistency": type_consistency,
        "expected_output_kind": expected_kind,
        "runtime_output_kind": runtime_kind,
        "ir_output_kind": ir_kind,
        "ir_type": ir_type,
        "unjustified_operations": unjustified,
        "items": [i.to_dict() for i in contract.items],
    }


def _stage_label(value) -> str:
    """yes / no for boolean stage coverage; NOT_RUN kept honest."""
    if value is None or value is False:
        return "no"
    if value == NOT_RUN:
        return "NOT_RUN"
    return "yes"


def format_coverage(matrix: dict) -> str:
    """A compact human-readable coverage report (CLI/trace)."""
    if not matrix:
        return ""
    lines = [
        "Requirement Coverage:",
    ]
    for item in matrix.get("items", []):
        if not item.get("required", True):
            continue
        stages = item.get("stages") or {}
        cells = " ".join(
            f"{name}={_stage_label(stages.get(name))}"
            for name in ("semantic", "plan", "ir", "code", "runtime",
                         "behavioral_test", "verified")
        )
        ev = (item.get("evidence") or "").strip()
        ev = f"  [{ev}]" if ev else ""
        lines.append(
            f"  {item['id']}: {item['semantic']} — {cells}{ev}")
    verdict = "PASS" if matrix.get("passed") else "FAIL"
    lines.append(
        f"  => {verdict} ({matrix.get('verified', 0)}/"
        f"{matrix.get('total', 0)} verified, "
        f"type={matrix.get('type_consistency')}"
        f"[{matrix.get('expected_output_kind')}/{matrix.get('runtime_output_kind')}])")
    return "\n".join(lines)
