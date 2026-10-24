"""Verification error analysis, repair planning, and repair execution.

The repair system is structural, not textual:

1. ErrorAnalyzer converts code + the required operation graph into
   structured errors (missing op, unreachable op, premature return,
   placeholder, behavioral mismatch, runtime error, syntax error).
2. FixPlanner converts structured errors into a repair plan over the
   operation/IR region (e.g. rebuild pipeline region, switch strategy
   variant for a misbehaving operation).
3. RepairEngine regenerates the affected code from the operation graph —
   never blind string edits.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass, field
from typing import Any, Optional

from brain.generation.op_semantics import (
    _OP_REGISTRY,
    UnknownOperationError,
    available_variants,
    build_pipeline,
    generated_test_inputs,
    reference_pipeline,
    is_direct_algorithm,
    _DIRECT_ALGORITHMS,
)


# --------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------- #

#: Structural markers for composed operations that have no standalone
#: pipeline template (they are rendered by the composer's generator):
#: the analyzer must still LOCATE them, or an honest implementation
#: reads as "missing operation".
_EXTRA_MARKERS: dict[str, list[str]] = {
    "count_frequency": ["Counter(", "freq"],
    "compare_equal_to": ["== 1", "==1", "== "],
    "select_first": ["[0] if"],
    "fallback_none": ["else None"],
}


class ErrorAnalyzer:
    """Produces structured verification errors from generated code."""

    def analyze(self, code: str, required_ops: list[str],
                contract=None) -> list[dict]:
        errors: list[dict] = []

        if not code or not code.strip():
            errors.append({"type": "EMPTY_OUTPUT",
                           "cause": "No code was generated"})
            return errors

        # 1. Syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append({
                "type": "SYNTAX_ERROR",
                "cause": str(e),
                "location": f"line {e.lineno}",
            })
            return errors

        funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        if not funcs:
            errors.append({"type": "MISSING_FUNCTION",
                           "cause": "No function definition found"})
            return errors
        func = funcs[0]
        # A helper defined BEFORE the artifact (copy_list before
        # sort_list): the REQUIRED OPERATION names the artifact
        # function — the analysis targets it, never the first
        # definition in the file.
        if required_ops:
            for candidate in funcs:
                if candidate.name in required_ops:
                    func = candidate
                    break

        # 2. Placeholder / stub operations
        src_lines = code.split("\n")
        for i, line in enumerate(src_lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_body = stripped.lstrip("#").strip().lower()
                for op in required_ops:
                    if comment_body == op or comment_body == op.replace("_", " "):
                        errors.append({
                            "type": "PLACEHOLDER_OPERATION",
                            "operation": op,
                            "location": f"line {i + 1}",
                            "cause": "Required operation emitted as comment stub",
                            "expected": f"executable code implementing {op}",
                            "actual": stripped,
                        })
        # Bare pass as the only real statement
        real_body = [s for s in func.body
                     if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
                     and not isinstance(s, ast.Pass)]
        if any(isinstance(s, ast.Pass) for s in func.body) and not real_body:
            errors.append({"type": "PLACEHOLDER_PASS",
                           "cause": "Function body contains only pass"})

        # 3. Premature return / unreachable operations
        return_idx = None
        for i, stmt in enumerate(func.body):
            if isinstance(stmt, ast.Return):
                return_idx = i
                break
        op_positions = self._locate_operations(func, src_lines, required_ops)
        for op in required_ops:
            pos = op_positions.get(op)
            if pos is None:
                # no executor line for this op at all
                if not any(e.get("operation") == op for e in errors):
                    errors.append({
                        "type": "MISSING_REQUIRED_OPERATION",
                        "operation": op,
                        "cause": "No executable code implements this operation",
                        "expected": f"executable step implementing {op}",
                        "actual": "absent",
                    })
            elif return_idx is not None and pos[0] > return_idx:
                errors.append({
                    "type": "UNREACHABLE_OPERATION",
                    "operation": op,
                    "location": f"line {pos[1]}",
                    "cause": "premature return",
                    "expected": f"{op} executes before the return statement",
                    "actual": f"{op} appears after return at statement {return_idx}",
                })

        # 4. Behavioral verification against reference semantics
        if required_ops:
            errors.extend(self._behavioral_errors(
                code, func.name, required_ops, contract))

        return errors

    def _locate_operations(self, func, src_lines, ops) -> dict[str, tuple[int, int]]:
        """Map op -> (body_stmt_index, source_line), using template markers."""
        positions: dict[str, tuple[int, int]] = {}
        search_lines = src_lines

        for idx, stmt in enumerate(func.body):
            seg_start = getattr(stmt, "lineno", 0)
            seg_end = getattr(stmt, "end_lineno", seg_start)
            seg_text = "\n".join(search_lines[max(0, seg_start - 1):seg_end])

            for op in ops:
                if op in positions:
                    continue
                # Check pipeline registry markers
                markers = _OP_REGISTRY.get(op, {}).get("markers", [])
                markers = list(markers) + _EXTRA_MARKERS.get(op, [])
                if any(m in seg_text or m in seg_text.replace(" ", "")
                       for m in markers):
                    positions[op] = (idx, seg_start)
                    continue
                # Check direct algorithm markers
                if op in _DIRECT_ALGORITHMS:
                    dmarkers = _DIRECT_ALGORITHMS[op].get("markers", [])
                    if any(m in seg_text or m in seg_text.replace(" ", "")
                           for m in dmarkers):
                        positions[op] = (idx, seg_start)
        return positions

    def _behavioral_errors(self, code: str, fname: str, ops: list[str],
                           contract=None) -> list[dict]:
        errors: list[dict] = []
        ns: dict = {}
        try:
            exec(code, ns)
        except Exception as e:
            errors.append({"type": "RUNTIME_ERROR",
                           "cause": f"{type(e).__name__}: {e}"})
            return errors

        func = ns.get(fname)
        if not callable(func):
            errors.append({"type": "MISSING_FUNCTION",
                           "cause": f"Function {fname} not callable"})
            return errors

        try:
            ref = reference_pipeline(ops, contract)
        except UnknownOperationError:
            return errors

        # Generate appropriate test inputs based on operation type
        test_cases = _generate_test_cases(ops, contract)

        for case in test_cases:
            expected, actual, error_msg = None, None, None
            try:
                if len(ops) == 1 and ops[0] in _DIRECT_ALGORITHMS:
                    # Direct algorithm: call with positional args
                    expected = ref(*case)
                else:
                    # Pipeline: call with data list
                    expected = ref(case)
            except Exception:
                continue
            try:
                if len(ops) == 1 and ops[0] in _DIRECT_ALGORITHMS:
                    actual = func(*case)
                else:
                    actual = func(list(case))
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors.append({
                    "type": "RUNTIME_ERROR",
                    "cause": error_msg,
                    "input": case,
                })
                return errors

            if not _values_equal(expected, actual):
                errors.append({
                    "type": "INCORRECT_RESULT",
                    "cause": "Generated behavior differs from semantic specification",
                    "input": case,
                    "expected": expected,
                    "actual": actual,
                })
                return errors
        return errors


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return list(a) == list(b)
    return a == b


def _generate_test_cases(ops: list[str], contract=None) -> list:
    """Generate semantically meaningful test inputs for the given ops.

    For direct algorithms: test cases are tuples of positional arguments.
    For pipeline operations: test cases are data lists.

    The THINK budget widens the search: a deeper request pays for
    adversarial inputs beyond the base set — inputs whose shape targets
    the pipeline's edge structure (post-filter emptiness, monotone runs,
    heavy duplication), never a particular problem sentence.
    """
    from brain.think_budget import current_budget
    budget = current_budget()
    if len(ops) == 1 and ops[0] in _DIRECT_ALGORITHMS:
        algo = _DIRECT_ALGORITHMS[ops[0]]
        n_params = algo["ref_params"]
        if ops[0] == "add_two_numbers":
            return [(3, 5), (0, 0), (-1, 1), (100, 200), (-3, -7)]
        elif ops[0] == "is_prime":
            return [(2,), (3,), (4,), (17,), (1,), (0,), (-5,)]
        elif ops[0] == "factorial":
            return [(0,), (1,), (5,), (10,)]
        elif ops[0] == "fibonacci":
            return [(0,), (1,), (6,), (10,)]
        elif ops[0] == "sum_list":
            return [([3, 1, 4, 1, 5],), ([0],), ([-5, 5],), ([],)]
        elif ops[0] == "average_list":
            return [([3, 1, 4, 1, 5],), ([10],), ([],)]
        elif ops[0] == "sort_list":
            return [([3, 1, 2],), ([],), ([1],), ([5, 5, 5],),
                    ([9, -1, 0, 4],)]
        elif ops[0] == "most_frequent_char":
            return [("hello",), ("aabbb",), ("x",)]
        else:
            return []
    # Pipeline operations: the base set, capped by what this request
    # paid for, then widened with adversarial inputs under think modes.
    cases = generated_test_inputs(contract)
    cases = cases[:max(budget.max_verification_tests, 1)]
    if budget.max_counterexamples:
        cases = cases + _adversarial_inputs(contract)[:budget.max_counterexamples]
    return cases


def _adversarial_inputs(contract=None) -> list:
    """Counterexample candidates derived from pipeline SHAPE, not from
    any problem: each input targets a structural edge — what remains
    after a filter stage empties the stream, a strictly monotone run
    (sorting already-sorted), heavy duplication (dedup idempotence)."""
    if contract is not None and (getattr(contract, "elem_type", "") or "") == "str":
        return [
            ["z"],                                   # single element
            ["b", "aa", "ccc", "dddd"],              # strictly growing lengths
            ["a"] * 12 + ["bb"],                     # heavy duplication
            ["one", "two", "three", "four"],         # varied lengths
            ["same", "same", "same", "other"],       # dominant duplicate
        ]
    return [
        [5],                        # single element: post-filter emptiness
        [0, -1, 1],                 # symmetric triple straddling zero
        list(range(-20, 21)),       # wide monotone run, zero mean
        [100, 50, 25, 12, 6, 3],    # strictly descending: sort stability
        [1] * 12 + [2],             # heavy duplication: dedup behavior
    ]


# --------------------------------------------------------------------- #
# Fix planning
# --------------------------------------------------------------------- #

@dataclass
class RepairPlan:
    """A structural repair plan derived from verification errors."""
    action: str                     # rebuild_pipeline | switch_variant
    ops: list[str] = field(default_factory=list)
    scaffold_variants: dict[str, str] = field(default_factory=dict)
    switch_ops: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.action == "switch_variant":
            return f"switch strategy variant for {self.switch_ops}"
        return f"rebuild pipeline region: {' -> '.join(self.ops)}"


class FixPlanner:
    """Converts structured verification errors into a repair plan."""

    def plan(self, errors: list[dict], ops: list[str],
             current_variants: dict[str, str]) -> Optional[RepairPlan]:
        if not errors:
            return None

        structural_types = {
            "SYNTAX_ERROR", "EMPTY_OUTPUT", "MISSING_FUNCTION",
            "PLACEHOLDER_OPERATION", "PLACEHOLDER_PASS",
            "MISSING_REQUIRED_OPERATION", "UNREACHABLE_OPERATION",
        }
        structures = [e for e in errors if e["type"] in structural_types]
        behavioral = [e for e in errors if e["type"] == "INCORRECT_RESULT"]

        if structures:
            bad_ops = [e.get("operation") for e in structures if e.get("operation")]
            return RepairPlan(
                action="rebuild_pipeline",
                ops=list(ops),
                notes=[f"{e['type']}: {e.get('operation') or e.get('cause')}"
                       for e in structures],
            )

        if behavioral:
            # pick an alternative variant for a suspicious operation
            for op in ops:
                current = current_variants.get(op)
                for alt in available_variants(op):
                    if alt != current:
                        nv = dict(current_variants)
                        nv[op] = alt
                        return RepairPlan(
                            action="switch_variant",
                            ops=list(ops),
                            scaffold_variants=nv,
                            switch_ops=[op],
                            notes=[f"Behavioral mismatch; retry {op} as {alt}"],
                        )
            return RepairPlan(action="rebuild_pipeline", ops=list(ops),
                              notes=["Behavioral mismatch; rebuild with defaults"])

        # Runtime errors without location info -> rebuild region
        return RepairPlan(action="rebuild_pipeline", ops=list(ops),
                          notes=[e.get("cause", "") for e in errors])


# --------------------------------------------------------------------- #
# Repair execution
# --------------------------------------------------------------------- #

class RepairEngine:
    """Applies repair plans by regenerating code from the operation graph."""

    def __init__(self):
        self.analyzer = ErrorAnalyzer()
        self.planner = FixPlanner()

    def analyze(self, code: str, ops: list[str], contract=None) -> list[dict]:
        return self.analyzer.analyze(code, ops, contract)

    def plan(self, errors: list[dict], ops: list[str],
             variants: dict[str, str]) -> Optional[RepairPlan]:
        return self.planner.plan(errors, ops, variants)

    def apply(self, plan: RepairPlan, func_name: str,
              contract=None) -> Optional[str]:
        """Regenerate code for the affected IR/dataflow region."""
        if plan is None:
            return None
        variants = plan.scaffold_variants if plan.action == "switch_variant" else None
        try:
            lines = build_pipeline(plan.ops, func_name,
                                   variants=variants, contract=contract)
        except UnknownOperationError:
            return None
        return "\n".join(lines)

    def verify(self, code: str, ops: list[str], contract=None) -> list[dict]:
        """Re-run analysis after repair; empty list means pass."""
        return self.analyzer.analyze(code, ops, contract)
