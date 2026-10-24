"""Typed program DSL, enumerative synthesis, and a learned macro library.

This is the experimental laboratory behind the growth loop.  Given
input/output examples, a bottom-up synthesizer searches a typed
expression space.  Candidates are deduplicated by their observational
behaviour on the training cases; a candidate is returned only after it
also passes held-out validation.  Nothing is "known" in advance about
which program solves the task.

Verified discoveries can be stored as *macros* — named typed
compositions with their own parameter signature.  Later synthesis can
reuse a macro as an operator, so a skill discovered in one context can
transfer to another without retraining.  Macros are only added by the
growth engine after independent verification.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from cognition.util import (atomic_write_json, read_json, stable_hash)
from cognition.vsa import VSA

NUM = "num"
LIST = "list"
BOOL = "bool"
STR = "str"


# --------------------------------------------------------------------- #
# Expression nodes
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Expr:
    op: str
    args: tuple = ()
    value: Any = None
    out_type: str = NUM

    def children(self) -> tuple:
        return self.args

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "out_type": self.out_type,
            "value": _json_value(self.value),
            "args": [a.to_dict() for a in self.args],
        }

    def describe(self) -> str:
        if self.op == "var":
            return str(self.value)
        if self.op == "const":
            return repr(self.value)
        if self.op == "macro":
            name = self.value.get("name") if isinstance(self.value, dict) else self.value
            return f"{name}({', '.join(a.describe() for a in self.args)})"
        return f"{self.op}({', '.join(a.describe() for a in self.args)})"

    def variables(self) -> list[str]:
        if self.op == "var":
            return [str(self.value)]
        out: list[str] = []
        for arg in self.args:
            for name in arg.variables():
                if name not in out:
                    out.append(name)
        return out


# (arg_types, return_type, evaluator, python_template, commutative)
Op = tuple[tuple[str, ...], str, Callable, str, bool]


def _safe_div(a, b):
    return a / b if b else 0.0


def _safe_mod(a, b):
    return a % b if b else 0


def _median(values):
    xs = sorted(values)
    n = len(xs)
    if not n:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if not values:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / len(values))


_OPS: dict[str, Op] = {
    # numeric
    "add": ((NUM, NUM), NUM, lambda a, b: a + b, "({} + {})", True),
    "sub": ((NUM, NUM), NUM, lambda a, b: a - b, "({} - {})", False),
    "mul": ((NUM, NUM), NUM, lambda a, b: a * b, "({} * {})", True),
    "div": ((NUM, NUM), NUM, _safe_div, "({} / {} if {} else 0.0)", False),
    "mod": ((NUM, NUM), NUM, _safe_mod, "({} % {} if {} else 0)", False),
    "max2": ((NUM, NUM), NUM, lambda a, b: max(a, b),
             "max({}, {})", True),
    "min2": ((NUM, NUM), NUM, lambda a, b: min(a, b),
             "min({}, {})", True),
    "neg": ((NUM,), NUM, lambda a: -a, "(-{})", False),
    "abs": ((NUM,), NUM, abs, "abs({})", False),
    "square": ((NUM,), NUM, lambda a: a * a, "({} * {})", False),
    "sign": ((NUM,), NUM, lambda a: (a > 0) - (a < 0), "(({} > 0) - ({} < 0))",
             False),
    "floor": ((NUM,), NUM, math.floor, "math.floor({})", False),
    "ceil": ((NUM,), NUM, math.ceil, "math.ceil({})", False),
    # list -> number
    "sum": ((LIST,), NUM, lambda xs: sum(xs), "sum({})", False),
    "prod": ((LIST,), NUM, lambda xs: math.prod(xs), "math.prod({})", False),
    "len": ((LIST,), NUM, len, "len({})", False),
    "max_list": ((LIST,), NUM, lambda xs: max(xs) if xs else 0,
                 "(max({}) if {} else 0)", False),
    "min_list": ((LIST,), NUM, lambda xs: min(xs) if xs else 0,
                 "(min({}) if {} else 0)", False),
    "mean": ((LIST,), NUM, _mean, "_mean({})", False),
    "median": ((LIST,), NUM, _median, "_median({})", False),
    "std": ((LIST,), NUM, _std, "_std({})", False),
    "first": ((LIST,), NUM, lambda xs: xs[0] if xs else 0,
              "({}[0] if {} else 0)", False),
    "last": ((LIST,), NUM, lambda xs: xs[-1] if xs else 0,
             "({}[-1] if {} else 0)", False),
    # list -> list
    "sort_asc": ((LIST,), LIST, sorted, "sorted({})", False),
    "sort_desc": ((LIST,), LIST, lambda xs: sorted(xs, reverse=True),
                  "sorted({}, reverse=True)", False),
    "reverse": ((LIST,), LIST, lambda xs: list(reversed(xs)),
                "list(reversed({}))", False),
    "unique": ((LIST,), LIST, lambda xs: list(dict.fromkeys(xs)),
               "list(dict.fromkeys({}))", False),
    "filter_even": ((LIST,), LIST,
                    lambda xs: [x for x in xs if x % 2 == 0],
                    "[x for x in {} if x % 2 == 0]", False),
    "filter_odd": ((LIST,), LIST,
                   lambda xs: [x for x in xs if x % 2 != 0],
                   "[x for x in {} if x % 2 != 0]", False),
    "filter_pos": ((LIST,), LIST,
                   lambda xs: [x for x in xs if x > 0],
                   "[x for x in {} if x > 0]", False),
    "filter_neg": ((LIST,), LIST,
                   lambda xs: [x for x in xs if x < 0],
                   "[x for x in {} if x < 0]", False),
    "map_square": ((LIST,), LIST, lambda xs: [x * x for x in xs],
                   "[x * x for x in {}]", False),
    "map_abs": ((LIST,), LIST, lambda xs: [abs(x) for x in xs],
                "[abs(x) for x in {}]", False),
    "map_neg": ((LIST,), LIST, lambda xs: [-x for x in xs],
                "[-x for x in {}]", False),
    "tail": ((LIST,), LIST, lambda xs: list(xs[1:]),
             "list({}[1:])", False),
    "init": ((LIST,), LIST, lambda xs: list(xs[:-1]),
             "list({}[:-1])", False),
    "range_len": ((LIST,), LIST, lambda xs: list(range(len(xs))),
                  "list(range(len({})))", False),
    "pairwise_add": ((LIST, LIST), LIST,
                     lambda a, b: [x + y for x, y in zip(a, b)],
                     "[x + y for x, y in zip({}, {})]", True),
    "pairwise_mul": ((LIST, LIST), LIST,
                     lambda a, b: [x * y for x, y in zip(a, b)],
                     "[x * y for x, y in zip({}, {})]", True),
    "dot": ((LIST, LIST), NUM,
            lambda a, b: sum(x * y for x, y in zip(a, b)),
            "sum(x * y for x, y in zip({}, {}))", True),
    # predicates
    "is_even": ((NUM,), BOOL, lambda x: x % 2 == 0, "({} % 2 == 0)", False),
    "is_odd": ((NUM,), BOOL, lambda x: x % 2 != 0, "({} % 2 != 0)", False),
    "is_pos": ((NUM,), BOOL, lambda x: x > 0, "({} > 0)", False),
    "is_neg": ((NUM,), BOOL, lambda x: x < 0, "({} < 0)", False),
    "is_zero": ((NUM,), BOOL, lambda x: x == 0, "({} == 0)", False),
    "is_empty": ((LIST,), BOOL, lambda xs: len(xs) == 0,
                 "(len({}) == 0)", False),
    "all_pos": ((LIST,), BOOL, lambda xs: all(x > 0 for x in xs),
                "all(x > 0 for x in {})", False),
    "any_even": ((LIST,), BOOL, lambda xs: any(x % 2 == 0 for x in xs),
                 "any(x % 2 == 0 for x in {})", False),
    "any_odd": ((LIST,), BOOL, lambda xs: any(x % 2 != 0 for x in xs),
                "any(x % 2 != 0 for x in {})", False),
    "lt": ((NUM, NUM), BOOL, lambda a, b: a < b, "({} < {})", False),
    "le": ((NUM, NUM), BOOL, lambda a, b: a <= b, "({} <= {})", False),
    "gt": ((NUM, NUM), BOOL, lambda a, b: a > b, "({} > {})", False),
    "ge": ((NUM, NUM), BOOL, lambda a, b: a >= b, "({} >= {})", False),
    "eq": ((NUM, NUM), BOOL, lambda a, b: a == b, "({} == {})", True),
    "and": ((BOOL, BOOL), BOOL, lambda a, b: a and b, "({} and {})", True),
    "or": ((BOOL, BOOL), BOOL, lambda a, b: a or b, "({} or {})", True),
    "not": ((BOOL,), BOOL, lambda a: not a, "(not {})", False),
    # conditionals (typed)
    "if_num": ((BOOL, NUM, NUM), NUM, lambda c, a, b: a if c else b,
               "({} if {} else {})", False),
    "if_list": ((BOOL, LIST, LIST), LIST, lambda c, a, b: a if c else b,
                "({} if {} else {})", False),
    "if_bool": ((BOOL, BOOL, BOOL), BOOL, lambda c, a, b: a if c else b,
                "({} if {} else {})", False),
    # strings
    "str_len": ((STR,), NUM, len, "len({})", False),
    "upper": ((STR,), STR, lambda s: s.upper(), "{}.upper()", False),
    "lower": ((STR,), STR, lambda s: s.lower(), "{}.lower()", False),
    "strip": ((STR,), STR, lambda s: s.strip(), "{}.strip()", False),
    "concat": ((STR, STR), STR, lambda a, b: a + b, "({} + {})", False),
    "capitalize": ((STR,), STR, lambda s: s.capitalize(),
                   "{}.capitalize()", False),
}


# --------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------- #
class EvaluationError(Exception):
    pass


def evaluate(expr: Expr, env: dict, macros: Optional[dict] = None) -> Any:
    if expr.op == "var":
        if expr.value not in env:
            raise EvaluationError(f"unbound variable {expr.value}")
        return env[expr.value]
    if expr.op == "const":
        return expr.value
    if expr.op == "macro":
        name = expr.value.get("name") if isinstance(expr.value, dict) else expr.value
        macro = (macros or {}).get(str(name))
        if macro is None:
            raise EvaluationError(f"unknown macro {name}")
        local = dict(env)
        for param, arg in zip(macro.params, expr.args):
            local[param] = evaluate(arg, env, macros)
        return evaluate(macro.body, local, macros)
    spec = _OPS.get(expr.op)
    if spec is None:
        raise EvaluationError(f"unknown op {expr.op}")
    args = [evaluate(arg, env, macros) for arg in expr.args]
    try:
        return spec[2](*args)
    except Exception as exc:
        raise EvaluationError(str(exc)) from exc


def evaluate_cases(expr: Expr, cases: Sequence[dict],
                   macros: Optional[dict] = None) -> Optional[tuple]:
    values = []
    for env in cases:
        try:
            values.append(evaluate(expr, env, macros))
        except Exception:
            return None
    return tuple(values)


def _json_value(value: Any) -> Any:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return str(value)


def _value_key(value: Any) -> str:
    try:
        return json.dumps(_json_value(value), sort_keys=True,
                          ensure_ascii=False)
    except Exception:
        return repr(value)


def _values_match(actual: tuple, expected: Sequence) -> bool:
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                return False
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=1e-7,
                                abs_tol=1e-7):
                return False
            continue
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    if not math.isclose(float(x), float(y), rel_tol=1e-7,
                                        abs_tol=1e-7):
                        return False
                elif x != y:
                    return False
            continue
        if a != b:
            return False
    return True


# --------------------------------------------------------------------- #
# Macros
# --------------------------------------------------------------------- #
@dataclass
class Macro:
    name: str
    params: list[str]
    param_types: list[str]
    body: Expr
    out_type: str
    domain: str = "general"
    uses: int = 0
    successes: int = 0
    created_at: float = field(default_factory=time.time)
    source_skill: str = ""
    kind: str = "macro"
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": list(self.params),
            "param_types": list(self.param_types),
            "body": self.body.to_dict(),
            "out_type": self.out_type,
            "domain": self.domain,
            "uses": self.uses,
            "successes": self.successes,
            "created_at": self.created_at,
            "source_skill": self.source_skill,
            "kind": self.kind,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Macro":
        return cls(
            name=str(data.get("name") or ""),
            params=[str(p) for p in data.get("params") or []],
            param_types=[str(t) for t in data.get("param_types") or []],
            body=expr_from_dict(data.get("body") or {}),
            out_type=str(data.get("out_type") or NUM),
            domain=str(data.get("domain") or "general"),
            uses=int(data.get("uses", 0)),
            successes=int(data.get("successes", 0)),
            created_at=float(data.get("created_at", time.time())),
            source_skill=str(data.get("source_skill") or ""),
            kind=str(data.get("kind") or "macro"),
            rationale=str(data.get("rationale") or ""),
        )


class ProgramLibrary:
    """Persistent library of verified compositional macros."""

    FILE = "program_library.json"

    def __init__(self, data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, self.FILE)
        self.macros: dict[str, Macro] = {}
        self._load()

    def add(self, params: list[str], param_types: list[str], body: Expr,
            domain: str = "general", source_skill: str = "") -> Macro:
        return self._add(params, param_types, body, domain=domain,
                         source_skill=source_skill, kind="macro",
                         prefix="m_")

    def add_primitive(self, params: list[str], param_types: list[str],
                      body: Expr, domain: str = "general",
                      source_skill: str = "",
                      rationale: str = "") -> Macro:
        return self._add(params, param_types, body, domain=domain,
                         source_skill=source_skill, kind="primitive",
                         prefix="p_", rationale=rationale)

    def _add(self, params: list[str], param_types: list[str],
             body: Expr, domain: str, source_skill: str, kind: str,
             prefix: str, rationale: str = "") -> Macro:
        signature = json.dumps(
            [params, param_types, body.to_dict()], sort_keys=True)
        for macro in self.macros.values():
            if macro.to_dict()["body"] == body.to_dict() and \
                    macro.param_types == param_types:
                return macro
        name = prefix + stable_hash(signature, length=10)
        macro = Macro(name=name, params=list(params),
                      param_types=list(param_types), body=body,
                      out_type=body.out_type, domain=domain,
                      source_skill=source_skill, kind=kind,
                      rationale=rationale)
        self.macros[name] = macro
        return macro

    def get(self, name: str) -> Optional[Macro]:
        return self.macros.get(str(name))

    def remove(self, name: str) -> bool:
        return self.macros.pop(str(name), None) is not None

    def all(self) -> list[Macro]:
        return sorted(self.macros.values(),
                      key=lambda m: (-m.successes, -m.uses, m.created_at))

    def for_types(self, param_types: list[str]) -> list[Macro]:
        return [m for m in self.all() if m.param_types == list(param_types)]

    def record_use(self, name: str, success: bool) -> None:
        macro = self.macros.get(name)
        if macro is None:
            return
        macro.uses += 1
        if success:
            macro.successes += 1

    def save(self) -> None:
        atomic_write_json(self.path, {
            "version": 1,
            "macros": [m.to_dict() for m in self.macros.values()],
        })

    def _load(self) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            return
        for row in data.get("macros") or []:
            try:
                macro = Macro.from_dict(row)
            except Exception:
                continue
            if macro.name:
                self.macros[macro.name] = macro

    def stats(self) -> dict:
        return {
            "macros": len(self.macros),
            "uses": sum(m.uses for m in self.macros.values()),
            "successes": sum(m.successes for m in self.macros.values()),
            "primitives": sum(1 for m in self.macros.values()
                              if m.kind == "primitive"),
            "domains": sorted({m.domain for m in self.macros.values()}),
        }


def expr_from_dict(data: dict) -> Expr:
    op = str(data.get("op") or "const")
    args = tuple(expr_from_dict(a) for a in data.get("args") or [])
    value = data.get("value")
    if op == "const" and isinstance(value, list):
        value = tuple(value)
    return Expr(op=op, args=args, value=value,
                out_type=str(data.get("out_type") or NUM))


# --------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------- #
@dataclass
class SynthesisResult:
    success: bool
    expression: Optional[Expr] = None
    code: str = ""
    train_accuracy: float = 0.0
    holdout_passed: bool = False
    candidates_generated: int = 0
    candidates_evaluated: int = 0
    matches: int = 0
    seconds: float = 0.0
    reason: str = ""
    macro_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "expression": self.expression.describe() if self.expression else "",
            "code": self.code,
            "train_accuracy": round(self.train_accuracy, 4),
            "holdout_passed": self.holdout_passed,
            "candidates_generated": self.candidates_generated,
            "candidates_evaluated": self.candidates_evaluated,
            "matches": self.matches,
            "seconds": round(self.seconds, 4),
            "reason": self.reason,
            "macro_used": list(self.macro_used),
        }


class Synthesizer:
    """Bottom-up typed enumerative synthesis with held-out validation."""

    def __init__(self, max_candidates: int = 20000,
                 max_cost: int = 6, variants_per_signature: int = 2):
        self.max_candidates = max(500, int(max_candidates))
        self.max_cost = max(2, int(max_cost))
        self.variants_per_signature = max(1, int(variants_per_signature))

    def synthesize(self, input_types: dict[str, str],
                   train_inputs: Sequence[dict], train_targets: Sequence,
                   holdout_inputs: Sequence[dict] = (),
                   holdout_targets: Sequence = (),
                   extra_constants: Iterable = (),
                   macros: Optional[dict[str, "Macro"]] = None
                   ) -> SynthesisResult:
        macros = dict(macros or {})
        started = time.time()
        if not train_inputs:
            return SynthesisResult(False, reason="no training cases",
                                   seconds=time.time() - started)
        targets = tuple(train_targets)
        target_type = _infer_target_type(targets, extra_constants)
        constants = self._constants(train_inputs, targets, extra_constants)
        libraries: dict[int, dict[str, list[tuple[Expr, tuple]]]] = {
            cost: {NUM: [], LIST: [], BOOL: [], STR: []}
            for cost in range(1, self.max_cost + 2)
        }
        seen: dict[tuple[str, str], int] = {}
        generated = 0
        evaluated = 0
        matches = 0
        best_failed_holdout: Optional[Expr] = None

        def add_candidate(expr: Expr, outputs: Optional[tuple], cost: int) -> bool:
            nonlocal generated, evaluated, matches, best_failed_holdout
            generated += 1
            if outputs is None:
                return False
            evaluated += 1
            signature = (expr.out_type, _value_key(outputs))
            count = seen.get(signature, 0)
            if count >= self.variants_per_signature:
                return False
            seen[signature] = count + 1
            libraries[cost][expr.out_type].append((expr, outputs))
            if expr.out_type == target_type and _values_match(outputs, targets):
                matches += 1
                holdout_ok = True
                if holdout_inputs:
                    holdout_output = evaluate_cases(
                        expr, holdout_inputs, macros)
                    holdout_ok = (holdout_output is not None
                                  and _values_match(holdout_output,
                                                     tuple(holdout_targets)))
                if holdout_ok:
                    return True
                if best_failed_holdout is None:
                    best_failed_holdout = expr
            return False

        # leaves
        for name, type_name in input_types.items():
            outputs = tuple(env.get(name) for env in train_inputs)
            if add_candidate(Expr("var", (), name, type_name), outputs, 1):
                expr = Expr("var", (), name, type_name)
                return self._success(expr, input_types, targets, True,
                                     generated, evaluated, matches, started,
                                     1, macros)
        for value in constants:
            type_name = _type_of(value)
            if add_candidate(Expr("const", (), value, type_name),
                             (value,) * len(train_inputs), 1):
                expr = Expr("const", (), value, type_name)
                return self._success(expr, input_types, targets, True,
                                     generated, evaluated, matches, started,
                                     0, macros)
        if target_type not in (NUM, LIST, BOOL, STR):
            return SynthesisResult(False, reason="unsupported target type",
                                   seconds=time.time() - started)

        for cost in range(2, self.max_cost + 1):
            for name, spec in _OPS.items():
                arg_types, out_type, _fn = spec[0], spec[1], spec[2]
                if out_type != target_type and cost >= self.max_cost:
                    continue
                for partition in _cost_partitions(cost - 1, len(arg_types)):
                    pools = []
                    for arg_type, part in zip(arg_types, partition):
                        pool = libraries.get(part, {}).get(arg_type, [])
                        if not pool:
                            pools = []
                            break
                        pools.append(pool)
                    if not pools:
                        continue
                    if spec[4] and len(arg_types) == 2:
                        pairs = _commutative_pairs(pools[0], pools[1])
                    else:
                        pairs = itertools.product(*pools)
                    for combo in pairs:
                        if generated >= self.max_candidates:
                            result = SynthesisResult(
                                False, expression=None,
                                candidates_generated=generated,
                                candidates_evaluated=evaluated,
                                matches=matches,
                                seconds=time.time() - started,
                                reason="candidate_budget")
                            if best_failed_holdout is not None:
                                result.expression = best_failed_holdout
                                result.reason = "candidate_budget_" \
                                                "holdout_failed"
                            return result
                        args = tuple(item[0] for item in combo)
                        expr = Expr(name, args, None, out_type)
                        outputs = _eval_outputs(expr, train_inputs, macros)
                        if add_candidate(expr, outputs, cost):
                            return self._success(expr, input_types, targets,
                                                 True, generated, evaluated,
                                                 matches, started, cost,
                                                 macros)
                # end partitions
            # macros are operators in the same typed space
            for macro in macros.values():
                arg_types = tuple(macro.param_types)
                if len(arg_types) > cost - 1:
                    continue
                for partition in _cost_partitions(cost - 1, len(arg_types)):
                    pools = []
                    for arg_type, part in zip(arg_types, partition):
                        pool = libraries.get(part, {}).get(arg_type, [])
                        if not pool:
                            pools = []
                            break
                        pools.append(pool)
                    if not pools:
                        continue
                    for combo in itertools.product(*pools):
                        if generated >= self.max_candidates:
                            break
                        args = tuple(item[0] for item in combo)
                        expr = Expr("macro", args,
                                    {"name": macro.name}, macro.out_type)
                        outputs = _eval_outputs(expr, train_inputs, macros)
                        if add_candidate(expr, outputs, cost):
                            return self._success(expr, input_types, targets,
                                                 True, generated, evaluated,
                                                 matches, started, cost,
                                                 macros)
            # end ops
        reason = "no_consistent_program"
        if best_failed_holdout is not None:
            reason = "holdout_failed"
        result = SynthesisResult(False, candidates_generated=generated,
                                 candidates_evaluated=evaluated,
                                 matches=matches,
                                 seconds=time.time() - started,
                                 reason=reason)
        result.expression = best_failed_holdout
        return result

    @staticmethod
    def _success(expr: Expr, input_types: dict, targets,
                 holdout_passed: bool, generated: int, evaluated: int,
                 matches: int, started: float, _cost: int,
                 macros: Optional[dict] = None) -> SynthesisResult:
        macros = macros or {}
        return SynthesisResult(
            True, expression=expr,
            code=expression_to_python(expr, input_types, macros=macros),
            train_accuracy=1.0, holdout_passed=holdout_passed,
            candidates_generated=generated,
            candidates_evaluated=evaluated, matches=matches,
            seconds=time.time() - started,
            reason="verified",
            macro_used=_collect_macro_names(expr))

    @staticmethod
    def _constants(inputs: Sequence[dict], targets: Sequence,
                   extra: Iterable) -> list:
        values: list = [0, 1, 2, -1, 0.5, 10]
        for case in inputs:
            for value in case.values():
                if isinstance(value, (int, float)) and not isinstance(value,
                                                                     bool):
                    values.extend([value, abs(value)])
                elif isinstance(value, str):
                    values.append(value)
        for target in targets:
            if isinstance(target, (int, float)) and not isinstance(target, bool):
                values.extend([target, abs(target)])
            elif isinstance(target, str):
                values.append(target)
        for value in extra:
            values.append(value)
        out = []
        for value in values:
            if isinstance(value, float) and not math.isfinite(value):
                continue
            if value not in out:
                out.append(value)
        return out[:18]


def _eval_outputs(expr: Expr, cases: Sequence[dict],
                  macros: Optional[dict] = None) -> Optional[tuple]:
    values = []
    for env in cases:
        try:
            values.append(evaluate(expr, env, macros))
        except Exception:
            return None
    return tuple(values)


def _collect_macro_names(expr: Expr) -> list[str]:
    out: list[str] = []
    if expr.op == "macro":
        name = expr.value.get("name") if isinstance(expr.value, dict) else expr.value
        if name and str(name) not in out:
            out.append(str(name))
    for arg in expr.args:
        for name in _collect_macro_names(arg):
            if name not in out:
                out.append(name)
    return out


def _commutative_pairs(left, right):
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            if left is not right or i <= j:
                yield (a, b)


def _cost_partitions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    if parts <= 0:
        if total == 0:
            yield ()
        return
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _cost_partitions(total - first, parts - 1):
            yield (first, *rest)


def _type_of(value: Any) -> str:
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, (int, float)):
        return NUM
    if isinstance(value, str):
        return STR
    if isinstance(value, (list, tuple)):
        return LIST
    return "any"


def _infer_target_type(targets: Sequence, extra: Iterable = ()) -> str:
    for value in targets:
        type_name = _type_of(value)
        if type_name != "any":
            return type_name
    for value in extra:
        type_name = _type_of(value)
        if type_name != "any":
            return type_name
    return NUM


# --------------------------------------------------------------------- #
# Python code generation
# --------------------------------------------------------------------- #
_PY_PREAMBLE = (
    "import math\n"
    "\n"
    "def _mean(xs):\n"
    "    return sum(xs) / len(xs) if xs else 0.0\n"
    "\n"
    "def _median(xs):\n"
    "    xs = sorted(xs)\n"
    "    n = len(xs)\n"
    "    if not n:\n"
    "        return 0.0\n"
    "    m = n // 2\n"
    "    return xs[m] if n % 2 else (xs[m - 1] + xs[m]) / 2.0\n"
    "\n"
    "def _std(xs):\n"
    "    if not xs:\n"
    "        return 0.0\n"
    "    m = _mean(xs)\n"
    "    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))\n"
    "\n"
)


def expression_to_python(expr: Expr, input_types: dict[str, str],
                         function_name: str = "solve",
                         macros: Optional[dict[str, Macro]] = None) -> str:
    params = list(input_types)
    body = _expr_to_python(expr, macros or {})
    signature = ", ".join(_safe_identifier(p) for p in params)
    return (_PY_PREAMBLE + f"def {function_name}({signature}):\n"
            f"    return {body}\n")


def _expr_to_python(expr: Expr, macros: dict) -> str:
    if expr.op == "var":
        return _safe_identifier(str(expr.value))
    if expr.op == "const":
        return repr(expr.value)
    if expr.op == "macro":
        name = expr.value.get("name") if isinstance(expr.value, dict) else expr.value
        macro = macros.get(str(name))
        if macro is None:
            return "0"
        local_macros = {k: v for k, v in macros.items() if k != name}
        env_names = {}
        for index, arg in enumerate(expr.args):
            env_names[macro.params[index]] = _expr_to_python(arg, macros)
        return _expr_to_python_with_env(macro.body, env_names, local_macros)
    spec = _OPS.get(expr.op)
    if spec is None:
        return "0"
    template = spec[3]
    args = [_expr_to_python(arg, macros) for arg in expr.args]
    if expr.op in ("div", "mod"):
        return template.format(args[0], args[1], args[1])
    if expr.op in ("max_list", "min_list", "first", "last"):
        return template.format(args[0], args[0])
    if expr.op == "sign":
        return template.format(args[0], args[0])
    if expr.op == "square":
        return template.format(args[0], args[0])
    return template.format(*args)


def _expr_to_python_with_env(expr: Expr, env: dict[str, str],
                             macros: dict) -> str:
    if expr.op == "var":
        return env.get(str(expr.value),
                       _expr_to_python(expr, macros))
    if expr.op in ("const", "macro"):
        return _expr_to_python(expr, macros)
    spec = _OPS.get(expr.op)
    if spec is None:
        return "0"
    template = spec[3]
    args = [_expr_to_python_with_env(arg, env, macros)
            for arg in expr.args]
    if expr.op in ("div", "mod"):
        return template.format(args[0], args[1], args[1])
    if expr.op in ("max_list", "min_list", "first", "last"):
        return template.format(args[0], args[0])
    if expr.op in ("sign", "square"):
        return template.format(args[0], args[0])
    return template.format(*args)


def _safe_identifier(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c == "_" else "_"
                      for c in str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = "v_" + cleaned
    return cleaned


def macro_op(macro: Macro):
    """Adapter exposing a Macro as a synthesizer operator."""
    def runner(*args):
        env = dict(zip(macro.params, args))
        return evaluate(macro.body, env, {})

    template = f"{macro.name}({{}})"
    return ((tuple(macro.param_types), macro.out_type, runner, "{}", False),
            macro)


def synthesized_macros(result: SynthesisResult,
                       input_types: dict[str, str],
                       library: ProgramLibrary,
                       domain: str = "general",
                       source_skill: str = "") -> Optional[Macro]:
    """Register a verified program as a reusable macro."""
    if not result.success or result.expression is None:
        return None
    return library.add(list(input_types), list(input_types.values()),
                       result.expression, domain=domain,
                       source_skill=source_skill)
