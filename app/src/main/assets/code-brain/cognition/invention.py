"""Invent new typed primitives by abstracting verified programs.

The inventor takes expressions that already survived the growth loop,
finds numeric constants or numeric sub-expressions, and replaces one of
them with a fresh numeric parameter.  The result is a new reusable
primitive ``p_...`` in the program library.  It is a genuine extension
of the typed vocabulary, but it is derived from existing operators; the
inventor does not create arbitrary Python code or a new physical sensor.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from cognition.config import GrowthConfig
from cognition.dsl import (BOOL, LIST, NUM, STR, Expr, Macro, evaluate,
                           expression_to_python, expr_from_dict)
from cognition.sandbox import Sandbox
from cognition.util import (atomic_write_json, json_safe, read_json,
                            stable_hash)


def _replace_at(expr: Expr, path: tuple[int, ...],
                replacement: Expr) -> Expr:
    if not path:
        return replacement
    index = path[0]
    args = list(expr.args)
    if index >= len(args):
        raise IndexError("expression path out of range")
    args[index] = _replace_at(args[index], path[1:], replacement)
    return Expr(expr.op, tuple(args), expr.value, expr.out_type)


def _paths(expr: Expr, path: tuple[int, ...] = ()):
    yield path, expr
    for index, child in enumerate(expr.args):
        yield from _paths(child, path + (index,))


def _values_close(left, right) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    if isinstance(left, (int, float)) and \
            isinstance(right, (int, float)):
        return math.isclose(float(left), float(right),
                            rel_tol=1e-7, abs_tol=1e-7)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _values_close(a, b) for a, b in zip(left, right))
    return left == right


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


@dataclass
class PrimitiveCandidate:
    body: Expr
    input_types: dict
    params: list[str]
    source: str
    abstracted: Expr
    rationale: str
    original: Expr
    status: str = "proposed"
    evidence: dict = field(default_factory=dict)
    name: str = ""

    def to_dict(self) -> dict:
        return {
            "body": json_safe(self.body.to_dict()),
            "body_text": self.body.describe(),
            "params": list(self.params),
            "input_types": dict(self.input_types),
            "source": self.source,
            "abstracted": json_safe(self.abstracted.to_dict()),
            "rationale": self.rationale,
            "status": self.status,
            "evidence": json_safe(self.evidence),
            "name": self.name,
        }


@dataclass
class PrimitiveInventionReport:
    session_id: str
    started_at: float
    cycles_requested: int
    cycles: int = 0
    candidates_generated: int = 0
    candidates_verified: int = 0
    candidates_adopted: int = 0
    primitives_before: int = 0
    primitives_after: int = 0
    candidates: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    stopped_reason: str = ""
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "cycles_requested": self.cycles_requested,
            "cycles": self.cycles,
            "candidates_generated": self.candidates_generated,
            "candidates_verified": self.candidates_verified,
            "candidates_adopted": self.candidates_adopted,
            "primitives_before": self.primitives_before,
            "primitives_after": self.primitives_after,
            "candidates": list(self.candidates),
            "duration_s": round(self.duration_s, 4),
            "stopped_reason": self.stopped_reason,
            "limitations": list(self.limitations),
        }


class PrimitiveInventor:
    def __init__(self, growth, config: Optional[GrowthConfig] = None,
                 seed: Optional[int] = None):
        self.growth = growth
        self.config = config or growth.config
        self.rng = random.Random(
            self.config.seed if seed is None else int(seed))
        self.library = growth.library
        self.sandbox = Sandbox(self.config)
        self.path = os.path.join(self.config.data_dir, "invention.json")

    def _evidence(self) -> list[tuple[Expr, dict, str]]:
        rows: list[tuple[Expr, dict, str]] = []
        seen = set()
        for macro in self.library.all():
            key = json.dumps(macro.body.to_dict(), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append((macro.body, dict(zip(macro.params,
                                              macro.param_types)),
                         "library:" + macro.name))
        skills = list(self.growth.skills.active()) + list(
            self.growth.skills.candidates())
        for skill in skills:
            payload = skill.payload or {}
            data = payload.get("expression")
            input_types = payload.get("input_types")
            if not isinstance(data, dict) or not isinstance(input_types, dict):
                continue
            try:
                expr = expr_from_dict(data)
            except Exception:
                continue
            key = json.dumps(expr.to_dict(), sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            rows.append((expr, dict(input_types),
                         "skill:" + str(skill.name)))
        return rows

    def _probes(self, input_types: dict) -> list[dict]:
        lists = ([], [1], [2, 4], [3, 1, 4, 1, 5], [-2, 0, 2],
                 [5, 5, 5], [0], [8, -3, 2, 7])
        probes = []
        for index in range(6):
            env = {}
            for name, type_name in input_types.items():
                if type_name == LIST:
                    env[name] = list(lists[index % len(lists)])
                elif type_name == NUM:
                    env[name] = [0.5, 1.0, 2.0, -1.0, 3.0, 1.5][
                        index % 6]
                elif type_name == BOOL:
                    env[name] = index % 2 == 0
                elif type_name == STR:
                    env[name] = ["abc", "", "Code 123", "x"][index % 4]
                else:
                    env[name] = 0
            probes.append(env)
        return probes

    def _abstract(self, expr: Expr, input_types: dict) -> list[dict]:
        proposals = []
        seen = set()
        if "k" in input_types:
            return proposals
        for path, node in _paths(expr):
            if not path or node.out_type != NUM:
                continue
            if node.op == "var":
                continue
            body = _replace_at(
                expr, path, Expr("var", (), "k", NUM))
            if "k" not in body.variables():
                continue
            free = body.variables()
            params = [name for name in input_types if name in free]
            if not params or set(free) - set(params) - {"k"}:
                continue
            params.append("k")
            types = [input_types[name] for name in params[:-1]] + [NUM]
            signature = json.dumps(
                [params, types, body.to_dict()], sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            kind = "constant" if node.op == "const" else "subexpression"
            proposals.append({
                "body": body,
                "input_types": {name: input_types[name]
                                for name in params[:-1]} | {"k": NUM},
                "params": params,
                "abstracted": node,
                "kind": kind,
                "source": "",
                "rationale": (
                    "replace a numeric %s in a verified expression "
                    "with a new numeric parameter" % kind),
            })
        return proposals

    @staticmethod
    def _foundation_evidence() -> list[tuple[Expr, dict, str]]:
        xs = Expr("var", (), "xs", LIST)
        total = Expr("sum", (xs,), None, NUM)
        length = Expr("len", (xs,), None, NUM)
        return [
            (Expr("div", (total, Expr("const", (), 2, NUM)),
                  None, NUM), {"xs": LIST}, "seed:sum_div"),
            (Expr("gt", (length, Expr("const", (), 3, NUM)),
                  None, BOOL), {"xs": LIST}, "seed:length_gt"),
            (Expr("mul", (length, Expr("const", (), 2, NUM)),
                  None, NUM), {"xs": LIST}, "seed:length_mul"),
        ]

    def _build_candidates(self) -> list[PrimitiveCandidate]:
        rows = []
        evidence = self._evidence() or self._foundation_evidence()
        for expression, input_types, source in evidence:
            for proposal in self._abstract(expression, input_types):
                proposal["source"] = source
                rows.append(PrimitiveCandidate(
                    body=proposal["body"],
                    input_types=proposal["input_types"],
                    params=proposal["params"],
                    source=source,
                    abstracted=proposal["abstracted"],
                    rationale=proposal["rationale"],
                    original=expression))
        if not rows:
            for expression, input_types, source in \
                    self._foundation_evidence():
                for proposal in self._abstract(expression, input_types):
                    proposal["source"] = source
                    rows.append(PrimitiveCandidate(
                        body=proposal["body"],
                        input_types=proposal["input_types"],
                        params=proposal["params"],
                        source=source,
                        abstracted=proposal["abstracted"],
                        rationale=proposal["rationale"],
                        original=expression))
        self.rng.shuffle(rows)
        return rows

    def _verify(self, candidate: PrimitiveCandidate) -> bool:
        probes = self._probes(
            {key: value for key, value in candidate.input_types.items()
             if key != "k"})
        if not probes:
            return False
        original_outputs = []
        hole_values = []
        for env in probes:
            try:
                original_outputs.append(
                    evaluate(candidate.original, env, self.library.macros))
                hole_values.append(
                    evaluate(candidate.abstracted, env,
                             self.library.macros))
            except Exception as exc:
                candidate.evidence = {
                    "reason": "original_evaluation: " +
                              type(exc).__name__ + ": " + str(exc)[:160]}
                return False
        for env, expected, hole in zip(probes, original_outputs,
                                       hole_values):
            local = dict(env)
            local["k"] = hole
            try:
                restored = evaluate(candidate.body, local,
                                    self.library.macros)
            except Exception as exc:
                candidate.evidence = {
                    "reason": "specialization: " +
                              type(exc).__name__ + ": " + str(exc)[:160]}
                return False
            if not _values_close(restored, expected):
                candidate.evidence = {
                    "reason": "specialization_mismatch"}
                return False
            if not _finite(restored):
                candidate.evidence = {"reason": "non_finite_specialization"}
                return False
        behaviors = []
        for value in (0.0, 1.0, -1.0, 2.0, 5.0, 0.5):
            for env in probes:
                local = dict(env)
                local["k"] = value
                try:
                    output = evaluate(candidate.body, local,
                                      self.library.macros)
                except Exception as exc:
                    candidate.evidence = {
                        "reason": "generalization: " +
                                  type(exc).__name__ + ": " +
                                  str(exc)[:160]}
                    return False
                if not _finite(output):
                    candidate.evidence = {"reason": "non_finite_generalization"}
                    return False
                behaviors.append(json.dumps(json_safe(output),
                                            sort_keys=True))
        if len(set(behaviors)) < 2:
            candidate.evidence = {"reason": "parameter_has_no_effect"}
            return False
        candidate.evidence = {
            "specialization_cases": len(probes),
            "generalization_cases": len(behaviors),
            "distinct_behaviors": len(set(behaviors)),
        }
        return True

    def _sandbox_verify(self, candidate: PrimitiveCandidate) -> bool:
        try:
            code = expression_to_python(
                candidate.body, candidate.input_types,
                function_name="invented",
                macros=self.library.macros)
            probes = self._probes(
                {key: value for key, value in
                 candidate.input_types.items() if key != "k"})
            cases = []
            for value in (0.0, 1.0, 2.0):
                for env in probes[:3]:
                    local = dict(env)
                    local["k"] = value
                    args = [local[name] for name in candidate.params]
                    expected = evaluate(candidate.body, local,
                                        self.library.macros)
                    cases.append({"args": args, "expected": expected})
            result = self.sandbox.verify_python_function(
                code, "invented", cases)
            if not result.ok:
                candidate.evidence["sandbox"] = (
                    result.reason or "; ".join(result.violations[-3:]))
            else:
                candidate.evidence["sandbox"] = "ok"
            return bool(result.ok)
        except Exception as exc:
            candidate.evidence["sandbox"] = (
                type(exc).__name__ + ": " + str(exc)[:160])
            return False

    def run(self, cycles: int = 1, candidates: int = 6,
            time_budget: Optional[float] = None) -> PrimitiveInventionReport:
        budget = float(time_budget if time_budget is not None else
                       max(1.0, float(getattr(
                           self.config, "invention_seconds", 20.0))))
        started = time.time()
        before = sum(1 for macro in self.library.all()
                     if macro.kind == "primitive")
        report = PrimitiveInventionReport(
            session_id="prim_" + stable_hash(
                time.time(), self.config.seed, length=12),
            started_at=started,
            cycles_requested=max(0, int(cycles)),
            primitives_before=before,
            limitations=[
                "Primitives are parametric abstractions of existing typed "
                "DSL expressions; this does not invent a new base operator "
                "or a physical sensor.",
                "Verification checks specialization, parameter variation and "
                "sandboxed code execution on generated probes, not external "
                "semantic truth.",
                "Candidates are bounded by the supplied evidence, cycle count "
                "and wall-clock budget; when no learned expression is "
                "available, a small foundation pattern set seeds the search.",
            ])
        proposals = self._build_candidates()
        if not proposals:
            report.candidates = []
            report.duration_s = time.time() - started
            report.stopped_reason = "no_evidence"
            self._save(report)
            return report
        limit = max(1, int(candidates))
        selected = proposals[:limit]
        report.candidates_generated = len(selected)
        for index in range(max(0, int(cycles))):
            if time.time() - started >= budget:
                report.stopped_reason = "time_budget"
                break
            report.cycles = index + 1
            for candidate in selected:
                if time.time() - started >= budget:
                    report.stopped_reason = "time_budget"
                    break
                if not self._verify(candidate):
                    candidate.status = "rejected"
                    report.candidates.append(candidate.to_dict())
                    continue
                candidate.status = "verified"
                report.candidates_verified += 1
                existing = next((
                    macro for macro in self.library.all()
                    if macro.to_dict()["body"] ==
                    candidate.body.to_dict() and
                    macro.param_types == list(
                        candidate.input_types.values())), None)
                if existing is not None:
                    candidate.status = "duplicate"
                    candidate.name = existing.name
                elif not self._sandbox_verify(candidate):
                    candidate.status = "rejected"
                else:
                    macro = self.library.add_primitive(
                        list(candidate.params),
                        list(candidate.input_types.values()),
                        candidate.body, domain="math",
                        rationale=candidate.rationale)
                    candidate.status = "adopted"
                    candidate.name = macro.name
                    report.candidates_adopted += 1
                    self.library.save()
                    self.growth.save()
                    self.growth.journal.log(
                        "primitive_invented", primitive=macro.name,
                        body=candidate.body.describe(),
                        source=candidate.source,
                        evidence=candidate.evidence)
                report.candidates.append(candidate.to_dict())
            break
        report.primitives_after = sum(
            1 for macro in self.library.all()
            if macro.kind == "primitive")
        report.duration_s = time.time() - started
        if not report.stopped_reason:
            report.stopped_reason = (
                "adopted" if report.candidates_adopted
                else "no_adopted_primitive")
        self._save(report)
        return report

    def _save(self, report: PrimitiveInventionReport) -> None:
        data = read_json(self.path, {})
        if not isinstance(data, dict):
            data = {}
        sessions = data.get("primitive_sessions")
        if not isinstance(sessions, list):
            sessions = []
        sessions.append(report.to_dict())
        data["primitive_sessions"] = sessions[-50:]
        data["last_primitive"] = report.to_dict()
        atomic_write_json(self.path, data)
