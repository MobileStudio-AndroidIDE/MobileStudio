"""Self-augmentation: recombine primitives into new verified skills.

This is deliberately narrower than the speculative "invent a new
modality" idea.  The engine can compose existing typed operators,
learned macros and verified program skills into expressions that never
appeared in the library, generate fresh verification tasks for them,
execute the generated Python independently in the sandbox, and adopt
only candidates that pass every gate.  Failed candidates never enter
the skill registry.

The verifier is generated from the candidate itself: fresh cases,
held-out cases and an independent code-execution check.  It validates
compositional behaviour and implementation, not external semantic
truth.  A candidate that is only a duplicate of an existing behaviour
is rejected by a probe suite.
"""
from __future__ import annotations

import itertools
import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from cognition.config import GrowthConfig
from cognition.domains import Candidate, Verification
from cognition.dsl import (BOOL, LIST, NUM, STR, _OPS, Expr, Macro,
                           evaluate, expression_to_python)
from cognition.growth import GrowthEngine
from cognition.skills import Skill
from cognition.util import (atomic_write_json, json_safe, read_json,
                            stable_hash)


def _op_names(arg_types: tuple, out_type: str) -> list[str]:
    rows = []
    for name, spec in _OPS.items():
        if tuple(spec[0]) == tuple(arg_types) and spec[1] == out_type:
            rows.append(name)
    return rows


_LIST_UNARY = _op_names((LIST,), LIST)
_LIST_BINARY = _op_names((LIST, LIST), LIST)
_LIST_TO_NUM = _op_names((LIST,), NUM)
_LIST_PREDICATE = _op_names((LIST,), BOOL)
_NUM_UNARY = _op_names((NUM,), NUM)
_NUM_BINARY = _op_names((NUM, NUM), NUM)
_NUM_COMPARE = [name for name in ("lt", "le", "gt", "ge", "eq")
                if name in _OPS]


def _call(op: str, args: list[Expr], out_type: str) -> Expr:
    return Expr(op, tuple(args), None, out_type)


def _macro_call(macro, args: list[Expr]) -> Expr:
    return Expr("macro", tuple(args), {"name": macro.name}, macro.out_type)


def _signature(expr: Expr) -> str:
    return json.dumps(expr.to_dict(), sort_keys=True)


def _primitive_names(expr: Expr) -> list[str]:
    rows: list[str] = []

    def walk(node: Expr) -> None:
        if node.op == "macro":
            name = (node.value or {}).get("name") \
                if isinstance(node.value, dict) else node.value
            rows.append(str(name))
        elif node.op not in ("var", "const"):
            rows.append(node.op)
        for child in node.args:
            walk(child)

    walk(expr)
    return sorted(set(rows))


def _node_count(expr: Expr) -> int:
    return 1 + sum(_node_count(child) for child in expr.args)


class _Candidate:
    def __init__(self, expression: Expr, input_types: dict,
                 source_primitives: list[str]):
        self.expression = expression
        self.input_types = dict(input_types)
        self.source_primitives = list(source_primitives)
        self.expression_data = expression.to_dict()
        self.id = "ac_" + stable_hash(_signature(expression),
                                     sorted(input_types.items()),
                                     length=12)
        self.description = expression.describe()
        self.target_type = expression.out_type
        self.code = ""
        self.verifier: dict = {}
        self.trials: list[dict] = []
        self.status = "proposed"
        self.skill_name = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "target_type": self.target_type,
            "source_primitives": list(self.source_primitives),
            "expression": json_safe(self.expression_data),
            "input_types": dict(self.input_types),
            "verifier": json_safe(self.verifier),
            "trials": json_safe(self.trials),
            "status": self.status,
            "skill_name": self.skill_name,
        }


@dataclass
class AugmentationReport:
    session_id: str
    started_at: float
    requested_cycles: int
    cycles: int = 0
    candidates_generated: int = 0
    candidates_verified: int = 0
    candidates_adopted: int = 0
    candidates_rejected: int = 0
    candidates_rolled_back: int = 0
    macros_before: int = 0
    macros_after: int = 0
    active_skills_before: int = 0
    active_skills_after: int = 0
    stopped_reason: str = ""
    duration_s: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["duration_s"] = round(float(data["duration_s"]), 4)
        return data


class SelfAugmentationEngine:
    """Bounded composition -> generated verifier -> adoption loop."""

    FILE = "augmentation.json"

    def __init__(self, growth: GrowthEngine,
                 config: Optional[GrowthConfig] = None,
                 seed: Optional[int] = None,
                 domain: str = "math"):
        self.growth = growth
        self.config = config or growth.config
        self.domain_name = str(domain or "math")
        self.domain = self.growth.domains.get(self.domain_name)
        if self.domain is None or not hasattr(self.domain,
                                              "task_from_expression"):
            for adapter in self.growth.domains.values():
                if hasattr(adapter, "task_from_expression"):
                    self.domain = adapter
                    self.domain_name = adapter.name
                    break
        if self.domain is None:
            raise ValueError("no compositional domain available")
        self.library = self.growth.library
        self.skills = self.growth.skills
        self.rng = random.Random(
            self.config.seed if seed is None else int(seed))
        self.path = os.path.join(self.config.data_dir, self.FILE)
        self.ledger = read_json(self.path, {})
        if not isinstance(self.ledger, dict):
            self.ledger = {}
        self._deadline = 0.0
        self.report: Optional[AugmentationReport] = None

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------
    def _macros(self) -> dict:
        return {macro.name: macro for macro in self.library.all()}

    def _existing_signatures(self) -> set[str]:
        rows = set()
        for macro in self.library.all():
            rows.add(_signature(macro.body))
        for skill in list(self.skills.active()) + list(
                self.skills.candidates()):
            data = (skill.payload or {}).get("expression")
            if isinstance(data, dict):
                try:
                    from cognition.dsl import expr_from_dict
                    rows.add(_signature(expr_from_dict(data)))
                except Exception:
                    continue
        return rows

    def _behavior_key(self, expr: Expr, envs: list[dict],
                      macros: dict) -> str:
        values = []
        for env in envs:
            try:
                values.append(json.dumps(evaluate(expr, env, macros),
                                         sort_keys=True, default=str))
            except Exception:
                values.append("__error__")
        return "|".join(values)

    def _probe_envs(self) -> list[dict]:
        rng = random.Random(self.config.seed + 404)
        rows = []
        for _ in range(5):
            length = rng.randint(4, 8)
            rows.append({"xs": [rng.randint(-5, 6) for _ in range(length)]})
        return rows

    def _existing_behavior(self, envs, macros) -> set[str]:
        rows = set()
        for macro in self.library.all():
            rows.add(self._behavior_key(macro.body, envs, macros))
        for skill in list(self.skills.active()) + list(
                self.skills.candidates()):
            data = (skill.payload or {}).get("expression")
            if not isinstance(data, dict):
                continue
            try:
                from cognition.dsl import expr_from_dict
                rows.add(self._behavior_key(
                    expr_from_dict(data), envs, macros))
            except Exception:
                continue
        return rows

    def _build_pool(self) -> list[_Candidate]:
        macros = self._macros()
        xs = Expr("var", (), "xs", LIST)
        macro_calls: list[tuple[Macro, Expr]] = []
        for macro in macros.values():
            options = []
            for type_name in macro.param_types:
                if type_name == LIST:
                    options.append([xs])
                elif type_name == NUM:
                    options.append([
                        Expr("const", (), value, NUM)
                        for value in (0, 1, 2)])
                elif type_name == BOOL:
                    options.append([Expr("const", (), value, BOOL)
                                    for value in (False, True)])
                elif type_name == STR:
                    options.append([Expr("const", (), "", STR)])
                else:
                    options = []
                    break
            if not options or len(options) > 3:
                continue
            for combo in itertools.product(*options):
                macro_calls.append((macro, _macro_call(
                    macro, list(combo))))
                if len(macro_calls) > 120:
                    break
            if len(macro_calls) > 120:
                break
        list_pool: list[Expr] = [xs]
        seen_list = {_signature(xs)}

        def add_list(expr: Expr) -> None:
            if len(list_pool) >= 90:
                return
            signature = _signature(expr)
            if signature in seen_list:
                return
            seen_list.add(signature)
            list_pool.append(expr)

        for op in _LIST_UNARY:
            add_list(_call(op, [xs], LIST))
        macro_lists: list[Expr] = []
        for macro, call in macro_calls:
            if macro.out_type == LIST:
                add_list(call)
                macro_lists.append(call)
        for expr in macro_lists:
            for op in _LIST_UNARY:
                add_list(_call(op, [expr], LIST))
        base_lists = list(list_pool[1:])
        for left in base_lists[:12]:
            for right in base_lists[:12]:
                if len(list_pool) >= 90:
                    break
                for op in _LIST_BINARY:
                    add_list(_call(op, [left, right], LIST))
        for expr in list(base_lists[:18]):
            for op in _LIST_UNARY:
                if op in ("tail", "init"):
                    continue
                add_list(_call(op, [expr], LIST))

        num_pool: list[Expr] = []
        seen_num = set()

        def add_num(expr: Expr) -> None:
            if len(num_pool) >= 120:
                return
            signature = _signature(expr)
            if signature in seen_num:
                return
            seen_num.add(signature)
            num_pool.append(expr)

        macro_nums: list[Expr] = []
        for macro, call in macro_calls:
            if macro.out_type == NUM:
                add_num(call)
                macro_nums.append(call)
        for expr in macro_nums:
            for op in _NUM_UNARY:
                add_num(_call(op, [expr], NUM))
        for expr in list_pool:
            for op in _LIST_TO_NUM:
                add_num(_call(op, [expr], NUM))
        base_nums = list(num_pool)
        for expr in base_nums[:24]:
            for op in _NUM_UNARY:
                add_num(_call(op, [expr], NUM))
        for left in base_nums[:14]:
            for right in base_nums[:14]:
                if len(num_pool) >= 120:
                    break
                for op in _NUM_BINARY:
                    add_num(_call(op, [left, right], NUM))

        bool_pool: list[Expr] = []
        for expr in list_pool:
            for op in _LIST_PREDICATE:
                bool_pool.append(_call(op, [expr], BOOL))
        for index, left in enumerate(num_pool[:24]):
            right = num_pool[(index + 1) % max(1, len(num_pool))]
            for op in _NUM_COMPARE:
                bool_pool.append(_call(op, [left, right], BOOL))
        macro_bools: list[Expr] = []
        for macro, call in macro_calls:
            if macro.out_type == BOOL:
                bool_pool.append(call)
                macro_bools.append(call)
        for expr in macro_bools:
            bool_pool.append(_call("not", [expr], BOOL))
            if bool_pool:
                bool_pool.append(_call("and", [expr, bool_pool[0]],
                                       BOOL))
        base_bools = list(bool_pool)
        for expr in base_bools[:20]:
            bool_pool.append(_call("not", [expr], BOOL))
        for index, left in enumerate(base_bools[:12]):
            right = base_bools[(index + 1) % max(1, len(base_bools))]
            bool_pool.append(_call("and", [left, right], BOOL))
            bool_pool.append(_call("or", [left, right], BOOL))

        existing = self._existing_signatures()
        candidates: list[_Candidate] = []
        seen_candidates = set()
        for expr in list_pool[1:] + num_pool + bool_pool:
            signature = _signature(expr)
            if signature in existing or signature in seen_candidates:
                continue
            if _node_count(expr) < 3:
                continue
            primitives = _primitive_names(expr)
            if len(primitives) < 2:
                continue
            if "xs" not in expr.variables():
                continue
            seen_candidates.add(signature)
            candidates.append(_Candidate(expr, {"xs": LIST}, primitives))
        self.rng.shuffle(candidates)
        macro_names = set(macros)
        candidates.sort(key=lambda row: (
            0 if any(name in macro_names
                     for name in row.source_primitives) else 1))
        return candidates

    def propose(self, limit: int = 8) -> list[_Candidate]:
        if not getattr(self.config, "augmentation_enabled", True):
            return []
        limit = max(1, int(limit))
        rows = self._build_pool()
        macros = self._macros()
        envs = self._probe_envs()
        known = self._existing_behavior(envs, macros)
        fresh = []
        for candidate in rows:
            key = self._behavior_key(candidate.expression, envs, macros)
            if key in known:
                continue
            candidate.verifier = {
                "kind": "generated_fresh_holdout_sandbox",
                "domain": self.domain_name,
                "trials": max(1, int(getattr(
                    self.config, "augmentation_trials", 3))),
                "checks": ["fresh_train", "holdout_accuracy",
                           "generated_code_execution",
                           "behavioural_novelty"],
                "acceptance": "all_trials_pass_and_gate_promotes",
                "probe_cases": len(envs),
            }
            fresh.append(candidate)
            if len(fresh) >= limit:
                break
        return fresh

    # ------------------------------------------------------------------
    # Verification and adoption
    # ------------------------------------------------------------------
    def _verify(self, candidate: _Candidate) -> bool:
        expression = candidate.expression
        input_types = dict(candidate.input_types)
        macros = self._macros()
        code = expression_to_python(expression, input_types,
                                    macros=macros)
        candidate.code = code
        payload = {
            "expression": expression.to_dict(),
            "input_types": dict(input_types),
            "code": code,
            "target_type": candidate.target_type,
            "macro_used": [name for name in candidate.source_primitives
                           if name in macros],
            "origin": "self_augmentation",
        }
        trials = max(1, int(getattr(self.config, "augmentation_trials", 3)))
        gate = self.skills.gate
        trials = max(trials, int(gate.min_successes),
                     int(gate.min_holdout))
        candidate.trials = []
        for index in range(trials):
            if self._deadline and time.monotonic() >= self._deadline:
                candidate.trials.append({"passed": False,
                                         "reason": "time_budget"})
                return False
            seed = (self.config.seed + 7919 * (index + 1) +
                    int(stable_hash(candidate.id, length=8), 16))
            rng = random.Random(seed)
            try:
                task = self.domain.task_from_expression(
                    expression, rng, difficulty=min(
                        0.9, 0.45 + 0.05 * index),
                    input_types=input_types, prefix="aug",
                    macros=macros)
            except Exception as exc:
                candidate.trials.append({
                    "passed": False,
                    "reason": "task_generation: " +
                              type(exc).__name__ + ": " + str(exc)[:200]})
                return False
            try:
                ctx = self.growth._context(2)
                generated = Candidate(
                    "program", dict(payload), description=candidate.description,
                    strategy="self_augmentation",
                    macro_names=list(payload["macro_used"]), cost=1)
                verification: Verification = self.domain.verify(
                    task, generated, ctx)
            except Exception as exc:
                candidate.trials.append({
                    "passed": False,
                    "reason": "verification: " +
                              type(exc).__name__ + ": " + str(exc)[:200]})
                return False
            row = {
                "trial": index + 1,
                "task": task.id,
                "passed": bool(verification.passed),
                "score": round(float(verification.score), 6),
                "checks": json_safe(verification.checks),
                "detail": verification.detail[:300],
            }
            candidate.trials.append(row)
            if not verification.passed:
                return False
        return bool(candidate.trials) and all(
            row.get("passed") for row in candidate.trials)

    def _adopt(self, candidate: _Candidate) -> bool:
        name = "aug_" + stable_hash(candidate.expression_data,
                                    candidate.input_types, length=10)
        existing = self.skills.find(name, self.domain_name)
        if existing is not None and existing.status == "active":
            candidate.status = "adopted"
            candidate.skill_name = existing.name
            return True
        before_macros = set(self.library.macros)
        skill: Optional[Skill] = None
        macro_name = ""
        try:
            payload = {
                "expression": candidate.expression.to_dict(),
                "input_types": dict(candidate.input_types),
                "code": candidate.code,
                "target_type": candidate.target_type,
                "macro_used": [item for item in candidate.source_primitives
                               if item in self.library.macros],
                "origin": "self_augmentation",
                "verifier": dict(candidate.verifier),
            }
            skill = self.skills.propose(
                name=name, domain=self.domain_name, kind="program",
                payload=payload,
                description=("self-augmented composition: " +
                             candidate.description)[:300],
                preconditions={"param_types": list(
                    candidate.input_types.values()),
                    "verifier": dict(candidate.verifier)},
                source_goal="self-augmentation")
            skill.capability_name = ("cognition.augmented." +
                                     name.replace(".", "_"))
            for row in candidate.trials:
                self.skills.record_attempt(
                    skill, True, holdout=True,
                    note=("self-augmentation trial " +
                          str(row.get("task", "")) + ": " +
                          str(row.get("detail", "")))[:300])
            if skill.status != "active":
                candidate.status = "rejected"
                candidate.skill_name = skill.name
                self.skills.save()
                self.library.save()
                return False
            macro = self.library.add(
                list(candidate.input_types),
                list(candidate.input_types.values()),
                candidate.expression, domain=self.domain_name,
                source_skill=skill.name)
            macro_name = macro.name
            candidate.status = "adopted"
            candidate.skill_name = skill.name
            skill.capability_name = skill.capability_name or (
                "cognition.augmented." + name.replace(".", "_"))
            self.skills.save()
            self.library.save()
            self.growth.save()
            self.growth.journal.log(
                "augmentation_adopted", domain=self.domain_name,
                skill=skill.name, macro=macro_name,
                primitives=candidate.source_primitives,
                trials=len(candidate.trials))
            return True
        except Exception as exc:
            if skill is not None:
                self.skills.retire(skill, "augmentation_rollback")
            if macro_name and macro_name not in before_macros:
                self.library.remove(macro_name)
            self.skills.save()
            self.library.save()
            candidate.status = "rolled_back"
            candidate.trials.append({
                "passed": False,
                "reason": "rollback: " + type(exc).__name__ + ": " +
                          str(exc)[:200]})
            self.growth.journal.log(
                "augmentation_rollback", domain=self.domain_name,
                skill=candidate.skill_name or name,
                error=f"{type(exc).__name__}: {exc}")
            return False

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def run(self, cycles: int = 1, candidates_per_cycle: int = 6,
            time_budget: Optional[float] = None) -> AugmentationReport:
        cycles = max(0, int(cycles))
        per_cycle = max(1, int(candidates_per_cycle))
        budget = float(time_budget if time_budget is not None else
                       max(1.0, float(getattr(
                           self.config, "augmentation_seconds", 20.0))))
        self._deadline = time.monotonic() + budget
        started = time.time()
        report = AugmentationReport(
            session_id="aug_" + stable_hash(time.time(), self.config.seed,
                                            length=12),
            started_at=started,
            requested_cycles=cycles,
            macros_before=len(self.library.macros),
            active_skills_before=len(self.skills.active()),
            limitations=[
                "Candidates are compositions of the existing typed "
                "operator/macro vocabulary; this engine does not invent a "
                "new primitive or a new sensor/actuator by itself.",
                "The generated verifier creates fresh tasks from the "
                "candidate composition and executes generated code in the "
                "sandbox. It checks compositional behaviour and code "
                "execution, not external real-world semantics.",
                "Adoption uses the same VerificationGate as growth: three "
                "successful held-out checks by default. Failed candidates "
                "are excluded and adoption exceptions are rolled back.",
                "Every daemon is bounded by cycles and wall-clock time.",
            ])
        self.report = report
        stopped = "cycles_completed"
        for cycle in range(1, cycles + 1):
            if time.monotonic() >= self._deadline:
                stopped = "time_budget"
                break
            report.cycles = cycle
            try:
                proposed = self.propose(limit=per_cycle)
            except Exception as exc:
                stopped = "candidate_generation_error"
                self.growth.journal.log(
                    "augmentation_generation_error",
                    error=f"{type(exc).__name__}: {exc}")
                break
            if not proposed:
                stopped = "no_candidates"
                break
            report.candidates_generated += len(proposed)
            for candidate in proposed:
                if time.monotonic() >= self._deadline:
                    stopped = "time_budget"
                    break
                ok = False
                try:
                    ok = self._verify(candidate)
                except Exception as exc:
                    candidate.status = "rejected"
                    candidate.trials.append({
                        "passed": False,
                        "reason": "unexpected: " + type(exc).__name__ +
                                  ": " + str(exc)[:200]})
                    self.growth.journal.log(
                        "augmentation_verify_error",
                        candidate=candidate.description,
                        error=f"{type(exc).__name__}: {exc}")
                if ok:
                    report.candidates_verified += 1
                    adopted = self._adopt(candidate)
                    if adopted:
                        report.candidates_adopted += 1
                    else:
                        report.candidates_rejected += 1
                else:
                    candidate.status = "rejected"
                    report.candidates_rejected += 1
                report.candidates.append(candidate.to_dict())
            self._save_ledger(report)
            if stopped in ("time_budget",):
                break
        report.stopped_reason = stopped
        report.macros_after = len(self.library.macros)
        report.active_skills_after = len(self.skills.active())
        report.duration_s = time.time() - started
        report.candidates = report.candidates[-200:]
        self._save_ledger(report)
        self.growth.journal.log(
            "augmentation_session", cycles=report.cycles,
            generated=report.candidates_generated,
            verified=report.candidates_verified,
            adopted=report.candidates_adopted,
            stopped=report.stopped_reason)
        return report

    def _save_ledger(self, report: Optional[AugmentationReport] = None) -> None:
        rows = self.ledger.get("sessions") or []
        if report is not None:
            rows.append(report.to_dict())
        candidates = list(self.ledger.get("candidates") or [])
        if report is not None:
            candidates.extend(report.candidates)
        candidates = candidates[-400:]
        atomic_write_json(self.path, {
            "version": 1,
            "sessions": rows[-50:],
            "candidates": candidates,
            "last_report": report.to_dict() if report else
            (self.ledger.get("last_report") or {}),
        })
        self.ledger = read_json(self.path, self.ledger)

    def status(self) -> dict:
        rows = self.ledger.get("candidates") or []
        return {
            "candidates": len(rows),
            "adopted": sum(1 for row in rows
                           if row.get("status") == "adopted"),
            "rejected": sum(1 for row in rows
                            if row.get("status") == "rejected"),
            "rolled_back": sum(1 for row in rows
                               if row.get("status") == "rolled_back"),
            "macros": len(self.library.macros),
            "active_skills": len(self.skills.active()),
            "last_report": self.ledger.get("last_report") or {},
        }
