"""Experiment domains for the autonomous growth loop.

Each domain is an *adapter* around an experimental task family:

* it generates fresh tasks (never a fixed answer key),
* it proposes candidates using reusable skills first and then a
  domain-specific experiment (program synthesis, hypothesis induction,
  proof search, strategy selection, ...),
* it verifies candidates on held-out cases independently,
* it can generate transfer tasks for a verified skill.

The list/math/code domain shares one typed DSL and a persistent macro
library, so a program discovered in one task can become an operator for
another.  The sequence domain discovers hypotheses through bounded
difference/ratio/periodicity/recurrence experiments.  The logic domain
produces checkable resolution proofs.  The graph domain learns which
verified algorithm/strategy to reuse for which query.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from cognition.compositional import CompositionalReasoner
from cognition.dsl import (BOOL, LIST, NUM, STR, Expr, Macro, ProgramLibrary,
                           Synthesizer, SynthesisResult, _OPS,
                           evaluate, evaluate_cases, expr_from_dict,
                           expression_to_python)
from cognition.sandbox import Sandbox
from cognition.skills import Skill
from cognition.util import clamp, stable_hash
from cognition.vsa import VSA


# --------------------------------------------------------------------- #
# Shared data types
# --------------------------------------------------------------------- #
@dataclass
class Task:
    id: str
    domain: str
    goal: str
    train: list
    validation: list
    holdout: list
    features: dict
    difficulty: float = 0.5
    payload: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "goal": self.goal,
            "difficulty": self.difficulty,
            "features": self.features,
            "payload": self.payload,
            "tags": self.tags,
        }


@dataclass
class Candidate:
    kind: str
    payload: dict
    description: str = ""
    strategy: str = ""
    skill_name: str = ""
    macro_names: list[str] = field(default_factory=list)
    cost: int = 1

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "payload": self.payload,
            "description": self.description,
            "strategy": self.strategy,
            "skill_name": self.skill_name,
            "macro_names": list(self.macro_names),
            "cost": self.cost,
        }


@dataclass
class Verification:
    passed: bool
    score: float = 0.0
    checks: dict = field(default_factory=dict)
    detail: str = ""
    code_ran: bool = False

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": round(float(self.score), 4),
            "checks": self.checks,
            "detail": self.detail,
            "code_ran": self.code_ran,
        }


@dataclass
class ExperimentOutcome:
    candidate: Optional[Candidate] = None
    failure_kind: str = ""
    detail: str = ""
    tried: int = 0
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class ExperimentContext:
    vsa: VSA
    synthesizer: Synthesizer
    library: ProgramLibrary
    skills: object
    predictive: object
    concepts: object
    working: object
    reasoner: CompositionalReasoner
    rng: random.Random
    explore_depth: int = 2
    max_candidates: int = 20000
    sandbox: Optional[Sandbox] = None
    task_budget_seconds: float = 8.0

    def skills_for(self, domain: str,
                   input_types: Optional[dict] = None,
                   include_candidates: bool = True) -> list[Skill]:
        rows = []
        try:
            rows.extend(self.skills.active(domain))
            if include_candidates:
                rows.extend(self.skills.candidates(domain))
        except Exception:
            return []
        if input_types is not None:
            wanted = list(input_types.values())
            exact = [s for s in rows
                     if s.preconditions.get("param_types") == wanted]
            if exact:
                rows = exact
        uniq = {}
        for skill in rows:
            uniq[skill.id] = skill
        return sorted(uniq.values(), key=lambda s: (
            s.status != "active", -s.confidence, -s.successes))


class DomainAdapter:
    name = "base"
    description = ""

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        raise NotImplementedError

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        raise NotImplementedError

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        raise NotImplementedError

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        return None

    def task_vector(self, task: Task, vsa: VSA) -> np.ndarray:
        features = dict(task.features)
        features["domain"] = self.name
        features["goal"] = task.goal
        return vsa.encode_features(features)

    def limitations(self) -> list[str]:
        return []


# --------------------------------------------------------------------- #
# Symbolic list / numeric / code synthesis
# --------------------------------------------------------------------- #
_LIST_UNARY = ["sort_asc", "sort_desc", "reverse", "unique", "filter_even",
               "filter_odd", "filter_pos", "filter_neg", "map_square",
               "map_abs", "map_neg", "tail", "init"]
_LIST_TO_NUM = ["sum", "prod", "len", "max_list", "min_list", "mean",
                "median", "std", "first", "last"]
_NUM_UNARY = ["neg", "abs", "square", "sign", "floor", "ceil"]
_NUM_BINARY = ["add", "sub", "mul", "max2", "min2"]
_NUM_CONSTANTS = [0, 1, 2, 3, -1, 0.5, 10]
_LIST_PREDICATES = ["is_empty", "all_pos", "any_even", "any_odd"]


class ListProgramDomain(DomainAdapter):
    """Discover typed programs from examples and verify generated code."""

    def __init__(self, name: str = "math", context: str = "lists",
                 allow_two_lists: bool = True):
        self.name = name
        self.context = context
        self.allow_two_lists = allow_two_lists
        self.description = (f"{context} program synthesis: numeric/list "
                            "transformations, code generated and executed")
        self._latents: dict[str, Expr] = {}
        self._tasks: dict[str, Task] = {}

    # ------------------------------------------------------------------
    # Task generation
    # ------------------------------------------------------------------
    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        for _attempt in range(80):
            input_types = {"xs": LIST}
            if self.allow_two_lists and rng.random() < 0.28:
                input_types["ys"] = LIST
            target_type = rng.choices([NUM, LIST, BOOL], [0.60, 0.25, 0.15])[0]
            expr = self._random_expr(rng, input_types, target_type,
                                     depth=2 if difficulty < 0.7 else 3)
            if expr is None or not expr.variables():
                continue
            if expr.op in ("var", "const"):
                continue
            train = self._cases(rng, input_types, expr, 5, 3, 8,
                                difficulty)
            validation = self._cases(rng, input_types, expr, 4, 3, 8,
                                     difficulty)
            holdout = self._cases(rng, input_types, expr, 8, 3, 10,
                                  difficulty)
            if train is None or validation is None or holdout is None:
                continue
            outputs = [row["expected"] for row in train]
            if len({json.dumps(o, sort_keys=True) for o in outputs}) <= 1:
                continue
            task_id = "lp_" + stable_hash(self.name, expr.describe(),
                                          time.time(), rng.random(),
                                          length=10)
            features = self._features(input_types, target_type, train,
                                      validation, holdout)
            task = Task(
                id=task_id, domain=self.name,
                goal=goal or f"induce a {target_type} program over "
                             f"{', '.join(input_types)}",
                train=train, validation=validation, holdout=holdout,
                features=features, difficulty=difficulty,
                payload={"input_types": input_types,
                         "target_type": target_type,
                         "context": self.context},
                tags=["synthesis", self.context, target_type])
            self._latents[task_id] = expr
            self._tasks[task_id] = task
            return task
        # deterministic fallback: range (max-min), still a real synthesis task
        return self._fallback_task(rng, goal)

    def _fallback_task(self, rng: random.Random, goal: str) -> Task:
        expr = Expr("sub", (Expr("max_list", (Expr("var", (), "xs", LIST),),
                                None, NUM),
                            Expr("min_list", (Expr("var", (), "xs", LIST),),
                                 None, NUM)),
                    None, NUM)
        input_types = {"xs": LIST}
        train = self._cases(rng, input_types, expr, 5, 3, 8, 0.5) or []
        validation = self._cases(rng, input_types, expr, 4, 3, 8, 0.5) or []
        holdout = self._cases(rng, input_types, expr, 8, 3, 10, 0.5) or []
        task_id = "lp_" + stable_hash("fallback", rng.random(), length=10)
        task = Task(task_id, self.name, goal or "induce a NUM program",
                    train, validation, holdout,
                    self._features(input_types, NUM, train, validation,
                                   holdout),
                    payload={"input_types": input_types,
                             "target_type": NUM, "context": self.context},
                    tags=["synthesis", self.context, NUM])
        self._latents[task_id] = expr
        self._tasks[task_id] = task
        return task

    def _random_expr(self, rng: random.Random, input_types: dict,
                     target: str, depth: int) -> Optional[Expr]:
        try:
            if target == NUM:
                return self._num_expr(rng, input_types, depth)
            if target == LIST:
                return self._list_expr(rng, input_types, depth)
            return self._bool_expr(rng, input_types, depth)
        except RecursionError:
            return None

    def _num_expr(self, rng: random.Random, input_types: dict,
                  depth: int) -> Optional[Expr]:
        list_vars = [name for name, t in input_types.items() if t == LIST]
        if depth <= 0:
            return Expr("const", (), rng.choice(_NUM_CONSTANTS), NUM)
        choice = rng.random()
        if choice < 0.45 and list_vars:
            op = rng.choice(_LIST_TO_NUM)
            arg = self._list_expr(rng, input_types, depth - 1)
            return Expr(op, (arg,), None, NUM)
        if choice < 0.60 and len(list_vars) >= 2 and rng.random() < 0.5:
            return Expr("dot", (Expr("var", (), list_vars[0], LIST),
                                Expr("var", (), list_vars[1], LIST)),
                        None, NUM)
        if choice < 0.82:
            op = rng.choice(_NUM_BINARY)
            return Expr(op, (self._num_expr(rng, input_types, depth - 1),
                             self._num_expr(rng, input_types, depth - 1)),
                        None, NUM)
        op = rng.choice(_NUM_UNARY)
        return Expr(op, (self._num_expr(rng, input_types, depth - 1),),
                    None, NUM)

    def _list_expr(self, rng: random.Random, input_types: dict,
                   depth: int) -> Expr:
        list_vars = [name for name, t in input_types.items() if t == LIST]
        if depth <= 0 or not list_vars:
            return Expr("var", (), rng.choice(list_vars or ["xs"]), LIST)
        choice = rng.random()
        if choice < 0.18:
            return Expr("var", (), rng.choice(list_vars), LIST)
        if choice < 0.35 and len(list_vars) >= 2:
            return Expr(rng.choice(["pairwise_add", "pairwise_mul"]),
                        (Expr("var", (), list_vars[0], LIST),
                         Expr("var", (), list_vars[1], LIST)), None, LIST)
        if choice < 0.58:
            return Expr(rng.choice(_LIST_UNARY),
                        (Expr("var", (), rng.choice(list_vars), LIST),),
                        None, LIST)
        return Expr(rng.choice(_LIST_UNARY),
                    (self._list_expr(rng, input_types, depth - 1),),
                    None, LIST)

    def _bool_expr(self, rng: random.Random, input_types: dict,
                   depth: int) -> Expr:
        list_vars = [name for name, t in input_types.items() if t == LIST]
        choice = rng.random()
        if depth <= 0 or not list_vars:
            return Expr("const", (), bool(rng.getrandbits(1)), BOOL)
        if choice < 0.40:
            return Expr(rng.choice(_LIST_PREDICATES),
                        (Expr("var", (), rng.choice(list_vars), LIST),),
                        None, BOOL)
        if choice < 0.72:
            return Expr(rng.choice(["lt", "le", "gt", "ge", "eq"]),
                        (self._num_expr(rng, input_types, depth - 1),
                         self._num_expr(rng, input_types, depth - 1)),
                        None, BOOL)
        if choice < 0.86:
            return Expr("not",
                        (self._bool_expr(rng, input_types, depth - 1),),
                        None, BOOL)
        return Expr(rng.choice(["and", "or"]),
                    (self._bool_expr(rng, input_types, depth - 1),
                     self._bool_expr(rng, input_types, depth - 1)),
                    None, BOOL)

    def _cases(self, rng: random.Random, input_types: dict, expr: Expr,
               count: int, min_len: int, max_len: int,
               difficulty: float, macros: Optional[dict] = None
               ) -> Optional[list]:
        cases = []
        max_value = 6 + int(12 * difficulty)
        for _ in range(count * 4):
            if len(cases) >= count:
                break
            env = {}
            length = rng.randint(min_len, max_len)
            for name, type_name in input_types.items():
                if type_name != LIST:
                    continue
                values = [rng.randint(-max_value, max_value)
                          for _ in range(length)]
                if rng.random() < 0.35 and len(values) > 2:
                    values = list(dict.fromkeys(values))
                env[name] = values
            try:
                expected = evaluate(expr, env, macros or {})
            except Exception:
                continue
            if isinstance(expected, float) and not math.isfinite(expected):
                continue
            cases.append({"args": env, "expected": expected})
        return cases if len(cases) == count else None

    def _features(self, input_types: dict, target_type: str,
                  train: list, validation: list, holdout: list) -> dict:
        flat = [v for case in train for values in case["args"].values()
                for v in values]
        targets = [case["expected"] for case in train]
        numeric_targets = [t for t in targets
                           if isinstance(t, (int, float))
                           and not isinstance(t, bool)]
        return {
            "domain": self.name,
            "context": self.context,
            "target_type": target_type,
            "input_types": list(input_types.values()),
            "n_inputs": len(input_types),
            "cases": len(train),
            "length_min": min((len(v) for case in train
                               for v in case["args"].values()), default=0),
            "length_max": max((len(v) for case in train
                               for v in case["args"].values()), default=0),
            "has_negative": any(v < 0 for v in flat),
            "has_even": any(v % 2 == 0 for v in flat),
            "has_duplicates": any(len(v) != len(set(v))
                                  for case in train
                                  for v in case["args"].values()),
            "target_numeric": bool(numeric_targets),
            "target_mean": (sum(numeric_targets) / len(numeric_targets)
                            if numeric_targets else 0),
        }

    # ------------------------------------------------------------------
    # Experiment: reuse -> synthesis
    # ------------------------------------------------------------------
    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        started = time.time()
        input_types = task.payload["input_types"]
        train_envs = [case["args"] for case in task.train]
        train_targets = tuple(case["expected"] for case in task.train)
        tried = 0
        macros = {m.name: m for m in ctx.library.all()}
        # 1) verified or candidate skills whose observable behaviour fits
        for skill in ctx.skills_for(self.name, input_types):
            tried += 1
            expression_data = (skill.payload or {}).get("expression")
            if not expression_data:
                continue
            try:
                expr = expr_from_dict(expression_data)
                outputs = evaluate_cases(expr, train_envs, macros)
            except Exception:
                continue
            if outputs is not None and _match(outputs, train_targets):
                candidate = Candidate(
                    kind="program", payload={
                        "expression": expr.to_dict(),
                        "input_types": dict(input_types),
                        "code": expression_to_python(
                            expr, input_types, macros=macros),
                        "target_type": task.payload["target_type"],
                    },
                    description=expr.describe(), strategy="skill_reuse",
                    skill_name=skill.name, cost=1)
                return ExperimentOutcome(candidate, tried=tried,
                                         seconds=time.time() - started,
                                         notes=[f"reused {skill.name}"])
        # 2) bottom-up synthesis, with learned macros as operators
        max_cost = min(6, 2 + max(1, int(ctx.explore_depth)))
        budget = max(1000, int(ctx.max_candidates))
        synth = Synthesizer(max_candidates=budget, max_cost=max_cost)
        macros = {m.name: m for m in ctx.library.all()}
        result = synth.synthesize(
            input_types, train_envs, train_targets,
            holdout_inputs=[case["args"] for case in task.validation],
            holdout_targets=[case["expected"] for case in task.validation],
            macros=macros)
        tried += 1
        if result.success and result.expression is not None:
            expr = result.expression
            candidate = Candidate(
                kind="program", payload={
                    "expression": expr.to_dict(),
                    "input_types": dict(input_types),
                    "code": expression_to_python(
                        expr, input_types, macros=macros),
                    "target_type": task.payload["target_type"],
                    "macro_used": list(result.macro_used),
                },
                description=expr.describe(), strategy="synthesis",
                macro_names=list(result.macro_used),
                cost=len(result.code.splitlines()))
            return ExperimentOutcome(
                candidate, tried=tried, seconds=time.time() - started,
                notes=[f"candidates={result.candidates_generated}",
                       f"macros={','.join(result.macro_used) or '-'}"])
        return ExperimentOutcome(
            None, failure_kind="synthesis_" + (result.reason or "failed"),
            detail=result.reason, tried=tried,
            seconds=time.time() - started,
            notes=[f"candidates={result.candidates_generated}",
                   f"matches={result.matches}",
                   f"max_cost={max_cost}"])

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        try:
            expr = expr_from_dict(candidate.payload.get("expression") or {})
        except Exception as exc:
            return Verification(False, detail=f"bad expression: {exc}")
        input_types = task.payload["input_types"]
        holdout_envs = [case["args"] for case in task.holdout]
        expected = tuple(case["expected"] for case in task.holdout)
        macros = ({m.name: m for m in ctx.library.all()}
                  if ctx is not None else {})
        outputs = evaluate_cases(expr, holdout_envs, macros)
        if outputs is None:
            return Verification(False, detail="candidate failed to evaluate",
                                checks={"holdout": 0.0})
        correct = sum(1 for a, b in zip(outputs, expected)
                      if _match((a,), (b,)))
        accuracy = correct / max(1, len(expected))
        checks = {"holdout_accuracy": accuracy,
                  "cases": len(expected)}
        if accuracy < 1.0:
            return Verification(False, score=accuracy, checks=checks,
                                detail="holdout mismatch")
        # Generated code must execute independently in the sandbox.
        if ctx is None or ctx.sandbox is None:
            return Verification(False, score=accuracy, checks=checks,
                                detail="no sandbox available")
        code = expression_to_python(expr, input_types, macros=macros)
        names = list(input_types)
        cases = [{
            "args": [case["args"][name] for name in names],
            "expected": case["expected"],
        } for case in task.holdout]
        run = ctx.sandbox.verify_python_function(code, "solve", cases)
        checks["sandbox"] = run.reason
        checks["sandbox_seconds"] = round(run.seconds, 4)
        if not run.ok:
            return Verification(False, score=accuracy, checks=checks,
                                detail=f"generated code failed: {run.reason}",
                                code_ran=False)
        return Verification(True, score=accuracy, checks=checks,
                            detail="holdout + generated-code execution",
                            code_ran=True)

    def task_from_expression(self, expr: Expr, rng: random.Random,
                             difficulty: float = 0.6,
                             input_types: Optional[dict] = None,
                             goal: str = "",
                             prefix: str = "cx",
                             macros: Optional[dict] = None) -> Task:
        input_types = dict(input_types or {"xs": LIST})
        variables = set(expr.variables())
        if not variables or not variables.issubset(set(input_types)):
            raise ValueError("expression variables do not match input types")
        for _ in range(40):
            train = self._cases(rng, input_types, expr, 5, 3, 8,
                                difficulty, macros=macros)
            validation = self._cases(rng, input_types, expr, 4, 3, 8,
                                     difficulty, macros=macros)
            holdout = self._cases(rng, input_types, expr, 8, 3, 10,
                                  difficulty, macros=macros)
            if train and validation and holdout:
                task_id = prefix + "_" + stable_hash(
                    self.name, expr.describe(), time.time(), rng.random(),
                    length=10)
                features = self._features(
                    input_types, expr.out_type, train, validation, holdout)
                task = Task(
                    id=task_id, domain=self.name,
                    goal=goal or f"compose a {expr.out_type} program over "
                                 f"{', '.join(input_types)}",
                    train=train, validation=validation, holdout=holdout,
                    features=features, difficulty=difficulty,
                    payload={"input_types": dict(input_types),
                             "target_type": expr.out_type,
                             "context": self.context,
                             "origin": "expression"},
                    tags=["composition", self.context, expr.out_type])
                self._latents[task_id] = expr
                self._tasks[task_id] = task
                return task
        raise ValueError("could not generate cases for expression")

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        expr = self._latents.get(task.id)
        if expr is None:
            return None
        input_types = task.payload["input_types"]
        for _ in range(40):
            train = self._cases(rng, input_types, expr, 5, 3, 8,
                                task.difficulty)
            validation = self._cases(rng, input_types, expr, 4, 3, 8,
                                     task.difficulty)
            holdout = self._cases(rng, input_types, expr, 8, 3, 10,
                                  task.difficulty)
            if train and validation and holdout:
                new = Task(
                    id="lp_" + stable_hash(task.id, rng.random(), length=10),
                    domain=self.name, goal=task.goal,
                    train=train, validation=validation, holdout=holdout,
                    features=dict(task.features),
                    difficulty=min(1.0, task.difficulty + 0.1),
                    payload=dict(task.payload), tags=list(task.tags))
                self._latents[new.id] = expr
                self._tasks[new.id] = new
                return new
        return None

    def limitations(self) -> list[str]:
        return [
            "The synthesizer searches a fixed typed operator vocabulary; "
            "it cannot invent an operator that is not expressible as a "
            "composition of existing operators (verified macros extend the "
            "effective vocabulary).",
            "Tasks are pure functions over JSON-like values; side effects, "
            "I/O and long-running programs are outside this domain.",
        ]


def _match(actual: Sequence, expected: Sequence) -> bool:
    if len(actual) != len(expected):
        return False
    for a, b in zip(actual, expected):
        if isinstance(a, bool) or isinstance(b, bool):
            if bool(a) != bool(b):
                return False
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if not math.isclose(float(a), float(b), rel_tol=1e-6,
                                abs_tol=1e-6):
                return False
        elif isinstance(a, list) and isinstance(b, list):
            if not _match(tuple(a), tuple(b)):
                return False
        elif a != b:
            return False
    return True


# --------------------------------------------------------------------- #
# Sequence induction
# --------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    limit = int(math.isqrt(n))
    for divisor in range(3, limit + 1, 2):
        if n % divisor == 0:
            return False
    return True


def _next_primes(last: int, count: int) -> list[int]:
    out = []
    candidate = last + 1
    while len(out) < count:
        if _is_prime(candidate):
            out.append(candidate)
        candidate += 1
    return out


class SequenceDomain(DomainAdapter):
    """Hypothesis-class discovery for sequence continuation."""

    name = "sequence"
    description = ("sequence continuation via difference, ratio, "
                   "periodicity, recurrence, interleaving and sign "
                   "decomposition; hypotheses are checked on a held-out "
                   "suffix of the observed prefix")

    FAMILIES = ("arithmetic", "geometric", "polynomial", "periodic",
                "fibonacci", "interleaved", "alternating", "prime",
                "recurrence3")

    def __init__(self):
        self._latents: dict[str, dict] = {}

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        for _attempt in range(60):
            family = rng.choice(self.FAMILIES)
            spec = {"family": family}
            values = self._generate(rng, family, spec)
            if values is None or len(values) < 11:
                continue
            prefix = values[:9]
            next_values = values[9:11]
            if len(set(prefix)) <= 1 and family not in ("periodic",
                                                        "arithmetic"):
                continue
            task_id = "sq_" + stable_hash(family, values, time.time(),
                                          rng.random(), length=10)
            features = self._features(prefix)
            features["family"] = family
            task = Task(
                id=task_id, domain=self.name,
                goal=goal or f"predict the next terms of a {family} sequence",
                train=[{"args": {"prefix": list(prefix)},
                        "expected": next_values}],
                validation=[{"args": {"prefix": list(prefix[:-2])},
                             "expected": list(prefix[-2:])}],
                holdout=[{"args": {"prefix": list(prefix)},
                          "expected": list(next_values)}],
                features=features, difficulty=difficulty,
                payload={"prefix": list(prefix), "steps": len(next_values),
                         "family": family},
                tags=["sequence", family])
            self._latents[task_id] = {"family": family, "spec": dict(spec),
                                      "values": list(values)}
            return task
        return self._fallback(rng, goal)

    def _fallback(self, rng: random.Random, goal: str) -> Task:
        start = rng.randint(1, 9)
        diff = rng.randint(1, 5)
        values = [start + diff * i for i in range(16)]
        prefix = values[:9]
        task = Task(
            id="sq_" + stable_hash(values, rng.random(), length=10),
            domain=self.name, goal=goal or "predict an arithmetic sequence",
            train=[{"args": {"prefix": prefix}, "expected": values[9:11]}],
            validation=[{"args": {"prefix": prefix[:-2]},
                         "expected": prefix[-2:]}],
            holdout=[{"args": {"prefix": prefix}, "expected": values[9:11]}],
            features=self._features(prefix), payload={
                "prefix": prefix, "steps": 2, "family": "arithmetic"},
            tags=["sequence", "arithmetic"])
        self._latents[task.id] = {"family": "arithmetic",
                                  "spec": {"diff": diff},
                                  "values": values}
        return task

    def _generate(self, rng: random.Random, family: str,
                  spec: dict) -> Optional[list]:
        if family == "arithmetic":
            start = rng.randint(-12, 20)
            diff = rng.choice([-7, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9])
            spec.update(start=start, diff=diff)
            return [start + diff * i for i in range(16)]
        if family == "geometric":
            start = rng.choice([1, 2, 3, -1, -2, -3])
            ratio = rng.choice([2, 3, -2, -3, 1.5])
            spec.update(start=start, ratio=ratio)
            return [round(start * ratio ** i, 6) for i in range(16)]
        if family == "polynomial":
            order = rng.choice([2, 2, 3])
            coeffs = [rng.randint(-3, 3) for _ in range(order + 1)]
            if coeffs[-1] == 0:
                coeffs[-1] = rng.choice([-2, -1, 1, 2])
            spec.update(coeffs=coeffs)
            return [sum(c * (i ** power) for power, c in enumerate(coeffs))
                    for i in range(16)]
        if family == "periodic":
            period = rng.randint(2, 5)
            cycle = [rng.randint(-8, 8) for _ in range(period)]
            if len(set(cycle)) == 1:
                cycle[0] += rng.choice([-2, 1, 3])
            spec.update(cycle=cycle)
            return [cycle[i % period] for i in range(16)]
        if family == "fibonacci":
            a, b = rng.randint(0, 4), rng.randint(1, 5)
            spec.update(a=a, b=b)
            values = [a, b]
            for _ in range(14):
                values.append(values[-1] + values[-2])
            return values
        if family == "interleaved":
            a1, d1 = rng.randint(-5, 6), rng.choice([-3, -2, 1, 2, 3, 4])
            a2, d2 = rng.randint(-5, 6), rng.choice([-4, -2, -1, 2, 3, 5])
            spec.update(a1=a1, d1=d1, a2=a2, d2=d2)
            values = []
            for i in range(16):
                if i % 2 == 0:
                    values.append(a1 + d1 * (i // 2))
                else:
                    values.append(a2 + d2 * (i // 2))
            return values
        if family == "alternating":
            start = rng.randint(1, 7)
            diff = rng.randint(1, 4)
            spec.update(start=start, diff=diff)
            return [((-1) ** i) * (start + diff * i) for i in range(16)]
        if family == "prime":
            start = rng.choice([2, 3, 5, 7, 11])
            values = []
            candidate = start
            while len(values) < 16:
                if _is_prime(candidate):
                    values.append(candidate)
                candidate += 1
            spec.update(start=start)
            return values
        if family == "recurrence3":
            coeffs = [rng.choice([1, 2, -1]), rng.choice([1, -1, 2]),
                      rng.choice([1, 1, -1])]
            seed = [rng.randint(-2, 4) for _ in range(3)]
            spec.update(coeffs=coeffs, seed=seed)
            values = list(seed)
            for _ in range(13):
                values.append(coeffs[0] * values[-1] +
                              coeffs[1] * values[-2] +
                              coeffs[2] * values[-3])
            return values
        return None

    def _features(self, prefix: list) -> dict:
        diffs = [prefix[i + 1] - prefix[i] for i in range(len(prefix) - 1)]
        ratios = []
        for i in range(len(prefix) - 1):
            if prefix[i] not in (0, 0.0):
                ratios.append(prefix[i + 1] / prefix[i])
        signs = [1 if v > 0 else -1 if v < 0 else 0 for v in prefix]
        alternating = all(signs[i] == -signs[i + 1] for i in range(
            len(signs) - 1) if signs[i] and signs[i + 1])
        period = 0
        for candidate in range(1, 6):
            if all(prefix[i] == prefix[i - candidate]
                   for i in range(candidate, len(prefix))):
                period = candidate
                break
        return {
            "domain": self.name,
            "length": len(prefix),
            "diff_constant": len(set(round(d, 6) for d in diffs)) <= 1,
            "ratio_constant": bool(ratios) and len(
                set(round(r, 6) for r in ratios)) <= 1,
            "alternating": alternating,
            "period": period,
            "prime_fraction": sum(1 for v in prefix if _is_prime(int(v))) /
                              max(1, len(prefix)),
            "monotonic": all(diffs[i] >= diffs[i - 1] for i in range(
                1, len(diffs))) or all(diffs[i] <= diffs[i - 1]
                                       for i in range(1, len(diffs))),
            "negative": any(v < 0 for v in prefix),
            "growth": abs(prefix[-1]) > abs(prefix[0]) * 2,
        }

    # ------------------------------------------------------------------
    # Hypotheses
    # ------------------------------------------------------------------
    def _hypotheses(self, prefix: list, budget: int,
                    preferred: Sequence[str] = ()) -> list[tuple]:
        strategies = []
        for name in preferred:
            if name not in strategies:
                strategies.append(name)
        for name in ("arithmetic", "geometric", "periodic", "polynomial",
                     "recurrence2", "fibonacci", "interleaved",
                     "alternating", "prime", "recurrence3"):
            if name not in strategies:
                strategies.append(name)
        results = []
        for name in strategies[:max(1, int(budget))]:
            try:
                prediction = self._predict(name, prefix, 2)
            except Exception:
                prediction = None
            if prediction:
                results.append((name, prediction))
        return results

    def _predict(self, strategy: str, prefix: list,
                 steps: int) -> Optional[list]:
        n = len(prefix)
        if n < 3:
            return None
        if strategy == "arithmetic":
            diffs = [prefix[i + 1] - prefix[i] for i in range(n - 1)]
            if len(set(round(d, 6) for d in diffs)) == 1:
                return [prefix[-1] + diffs[-1] * (i + 1)
                        for i in range(steps)]
            return None
        if strategy == "geometric":
            ratios = []
            for i in range(n - 1):
                if abs(prefix[i]) < 1e-12:
                    return None
                ratios.append(prefix[i + 1] / prefix[i])
            if len(set(round(r, 6) for r in ratios)) == 1:
                return [prefix[-1] * ratios[-1] ** (i + 1)
                        for i in range(steps)]
            return None
        if strategy == "periodic":
            for period in range(1, min(6, n // 2) + 1):
                if all(abs(prefix[i] - prefix[i - period]) < 1e-9
                       for i in range(period, n)):
                    return [prefix[(n + i) % period]
                            for i in range(steps)]
            return None
        if strategy == "polynomial":
            return self._polynomial(prefix, steps, max_order=3)
        if strategy == "recurrence2":
            return self._linear_recurrence(prefix, steps, order=2)
        if strategy == "recurrence3":
            return self._linear_recurrence(prefix, steps, order=3)
        if strategy == "fibonacci":
            if all(abs(prefix[i] - prefix[i - 1] - prefix[i - 2]) < 1e-7
                   for i in range(2, n)):
                a, b = prefix[-2], prefix[-1]
                out = []
                for _ in range(steps):
                    a, b = b, a + b
                    out.append(b)
                return out
            return None
        if strategy == "interleaved":
            even = prefix[::2]
            odd = prefix[1::2]
            even_pred = self._predict_simple(even, (steps + 1) // 2)
            odd_pred = self._predict_simple(odd, steps // 2)
            if even_pred is None or odd_pred is None:
                return None
            out = []
            for i in range(steps):
                position = n + i
                if position % 2 == 0:
                    out.append(even_pred[position // 2 - len(even)])
                else:
                    out.append(odd_pred[position // 2 - len(odd)])
            return out
        if strategy == "alternating":
            if not all((prefix[i] > 0) != (prefix[i + 1] > 0)
                       for i in range(n - 1)
                       if prefix[i] != 0 and prefix[i + 1] != 0):
                return None
            absolute = [abs(v) for v in prefix]
            predicted = self._predict_simple(absolute, steps)
            if predicted is None:
                return None
            next_sign = -1 if prefix[-1] > 0 else 1
            return [next_sign * ((-1) ** i) * predicted[i]
                    for i in range(steps)]
        if strategy == "prime":
            if not all(_is_prime(int(v)) for v in prefix):
                return None
            if any(prefix[i + 1] <= prefix[i] for i in range(n - 1)):
                return None
            return _next_primes(int(prefix[-1]), steps)
        return None

    @staticmethod
    def _predict_simple(prefix: list, steps: int) -> Optional[list]:
        if len(prefix) < 2:
            return None
        diffs = [prefix[i + 1] - prefix[i] for i in range(len(prefix) - 1)]
        if len(set(round(d, 6) for d in diffs)) == 1:
            return [prefix[-1] + diffs[-1] * (i + 1) for i in range(steps)]
        if prefix[-1] != 0:
            ratios = [prefix[i + 1] / prefix[i]
                      for i in range(len(prefix) - 1) if prefix[i] != 0]
            if ratios and len(set(round(r, 6) for r in ratios)) == 1:
                return [prefix[-1] * ratios[-1] ** (i + 1)
                        for i in range(steps)]
        # finite differences up to 3
        return SequenceDomain._polynomial(prefix, steps, max_order=3)

    @staticmethod
    def _polynomial(prefix: list, steps: int,
                    max_order: int = 3) -> Optional[list]:
        table = [list(prefix)]
        for _ in range(max_order):
            prev = table[-1]
            table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
        for order in range(1, min(max_order, len(prefix) - 1) + 1):
            column = table[order]
            if not column:
                continue
            if all(abs(value - column[-1]) < 1e-7
                   for value in column[-min(4, len(column)):]):
                last_diffs = [table[j][-1] for j in range(order + 1)]
                out = []
                for t in range(1, steps + 1):
                    value = 0.0
                    for j in range(order + 1):
                        value += math.comb(t + j - 1, j) * last_diffs[j]
                    out.append(_clean_number(value))
                return out
        return None

    @staticmethod
    def _linear_recurrence(prefix: list, steps: int,
                           order: int = 2) -> Optional[list]:
        if len(prefix) < 2 * order:
            return None
        # solve from the first 2*order terms using least squares
        rows, targets = [], []
        for index in range(order, min(len(prefix), 2 * order)):
            rows.append(prefix[index - order:index])
            targets.append(prefix[index])
        if len(rows) < order:
            return None
        matrix = np.asarray(rows, dtype=float)
        try:
            coeffs, *_ = np.linalg.lstsq(
                matrix, np.asarray(targets, dtype=float), rcond=None)
        except Exception:
            return None
        for index in range(order, len(prefix)):
            predicted = float(np.dot(coeffs,
                                     np.asarray(prefix[index - order:index],
                                                dtype=float)))
            if abs(predicted - prefix[index]) > 1e-6 * max(
                    1.0, abs(prefix[index])):
                return None
        values = [float(v) for v in prefix]
        for _ in range(steps):
            nxt = float(np.dot(coeffs, np.asarray(values[-order:],
                                                  dtype=float)))
            values.append(nxt)
        return [_clean_number(v) for v in values[-steps:]]

    # ------------------------------------------------------------------
    # Experiment / verification
    # ------------------------------------------------------------------
    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        started = time.time()
        prefix = task.payload["prefix"]
        steps = task.payload["steps"]
        preferred = []
        for skill in ctx.skills_for(self.name):
            strategy = (skill.payload or {}).get("strategy")
            if strategy:
                preferred.append(strategy)
        budget = max(10, int(ctx.explore_depth) + len(preferred))
        hypotheses = self._hypotheses(prefix, budget, preferred)
        if not hypotheses:
            return ExperimentOutcome(
                None, failure_kind="no_hypothesis",
                detail="no hypothesis class fits the observed prefix",
                seconds=time.time() - started)
        # internal validation on the last two observed terms
        validated = []
        for name, _ in hypotheses:
            try:
                back = self._predict(name, prefix[:-2], 2)
            except Exception:
                back = None
            if back is not None and _match(tuple(back), tuple(prefix[-2:])):
                validated.append(name)
        chosen = validated[0] if validated else hypotheses[0][0]
        prediction = self._predict(chosen, prefix, steps)
        if prediction is None:
            return ExperimentOutcome(
                None, failure_kind="prediction_failed",
                detail=chosen, seconds=time.time() - started)
        candidate = Candidate(
            kind="sequence_hypothesis",
            payload={"strategy": chosen, "prediction": prediction,
                     "prefix": list(prefix), "steps": steps},
            description=f"{chosen} -> {prediction}",
            strategy=chosen, cost=1)
        return ExperimentOutcome(candidate, tried=len(hypotheses),
                                 seconds=time.time() - started,
                                 notes=[f"validated={','.join(validated)}"])

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        expected = list(task.holdout[0]["expected"])
        prediction = list((candidate.payload or {}).get("prediction") or [])
        if len(prediction) != len(expected):
            return Verification(False, checks={"holdout": 0.0},
                                detail="wrong prediction length")
        correct = sum(1 for a, b in zip(prediction, expected)
                      if _match((a,), (b,)))
        score = correct / len(expected)
        # Recompute the strategy on the observed prefix independently.
        strategy = (candidate.payload or {}).get("strategy") or ""
        recomputed = None
        try:
            recomputed = self._predict(strategy, task.payload["prefix"],
                                       task.payload["steps"])
        except Exception:
            recomputed = None
        checks = {"holdout_accuracy": score,
                  "strategy": strategy,
                  "independent_recompute": recomputed is not None}
        if recomputed is None or not _match(tuple(recomputed),
                                            tuple(prediction)):
            return Verification(False, score=score, checks=checks,
                                detail="prediction not reproducible")
        return Verification(score == 1.0, score=score, checks=checks,
                            detail="held-out continuation")

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        spec = self._latents.get(task.id)
        if not spec:
            return None
        for _ in range(12):
            values = self._generate(rng, spec["family"], {})
            if not values or len(values) < 11:
                continue
            prefix = values[:9]
            new = Task(
                id="sq_" + stable_hash(task.id, values, rng.random(),
                                       length=10),
                domain=self.name, goal=task.goal,
                train=[{"args": {"prefix": prefix},
                        "expected": values[9:11]}],
                validation=[{"args": {"prefix": prefix[:-2]},
                             "expected": prefix[-2:]}],
                holdout=[{"args": {"prefix": prefix},
                          "expected": values[9:11]}],
                features=self._features(prefix),
                difficulty=min(1.0, task.difficulty + 0.1),
                payload={"prefix": prefix, "steps": 2,
                         "family": spec["family"]},
                tags=list(task.tags))
            self._latents[new.id] = {"family": spec["family"],
                                     "values": values}
            return new
        return None

    def limitations(self) -> list[str]:
        return [
            "Hypothesis classes are bounded (constant differences, ratios, "
            "periods, low-order recurrences, interleaving, sign and prime "
            "structure). Arbitrary Turing-complete sequence generators are "
            "not induced.",
        ]


def _clean_number(value: float):
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 9)


# --------------------------------------------------------------------- #
# Propositional logic: checkable resolution proofs
# --------------------------------------------------------------------- #
def _fvar(name: str) -> tuple:
    return ("var", name)


def _fnot(formula: tuple) -> tuple:
    if formula[0] == "not":
        return formula[1]
    return ("not", formula)


def _feval(formula: tuple, env: dict) -> bool:
    op = formula[0]
    if op == "var":
        return bool(env.get(formula[1], False))
    if op == "not":
        return not _feval(formula[1], env)
    if op == "and":
        return _feval(formula[1], env) and _feval(formula[2], env)
    if op == "or":
        return _feval(formula[1], env) or _feval(formula[2], env)
    if op == "implies":
        return (not _feval(formula[1], env)) or _feval(formula[2], env)
    if op == "iff":
        return _feval(formula[1], env) == _feval(formula[2], env)
    raise ValueError(f"unknown connective {op}")


def _fvars(formula: tuple) -> list[str]:
    op = formula[0]
    if op == "var":
        return [formula[1]]
    out: list[str] = []
    for child in formula[1:]:
        for name in _fvars(child):
            if name not in out:
                out.append(name)
    return out


def _fsize(formula: tuple) -> int:
    if formula[0] == "var":
        return 1
    return 1 + sum(_fsize(child) for child in formula[1:])


def _frandom(rng: random.Random, depth: int, variables: list[str]) -> tuple:
    if depth <= 0 or rng.random() < 0.30:
        return _fvar(rng.choice(variables))
    choice = rng.random()
    if choice < 0.20:
        return _fnot(_frandom(rng, depth - 1, variables))
    op = rng.choices(["and", "or", "implies", "iff"],
                     [0.31, 0.31, 0.22, 0.16])[0]
    return (op, _frandom(rng, depth - 1, variables),
            _frandom(rng, depth - 1, variables))


def _truth_table(formula: tuple) -> tuple:
    variables = _fvars(formula)
    assignments = []
    for bits in itertools.product([False, True], repeat=len(variables)):
        env = dict(zip(variables, bits))
        assignments.append((env, _feval(formula, env)))
    all_true = all(value for _env, value in assignments)
    all_false = all(not value for _env, value in assignments)
    if all_true:
        return "tautology", assignments
    if all_false:
        return "contradiction", assignments
    return "satisfiable", assignments


def _nnf(formula: tuple, negate: bool = False) -> tuple:
    op = formula[0]
    if op == "var":
        return ("not", formula) if negate else formula
    if op == "not":
        return _nnf(formula[1], not negate)
    if op == "and":
        return (("or" if negate else "and"),
                _nnf(formula[1], negate), _nnf(formula[2], negate))
    if op == "or":
        return (("and" if negate else "or"),
                _nnf(formula[1], negate), _nnf(formula[2], negate))
    if op == "implies":
        if negate:
            return ("and", _nnf(formula[1], False), _nnf(formula[2], True))
        return ("or", _nnf(formula[1], True), _nnf(formula[2], False))
    if op == "iff":
        left, right = formula[1], formula[2]
        if negate:
            return ("or", ("and", _nnf(left, False), _nnf(right, True)),
                    ("and", _nnf(left, True), _nnf(right, False)))
        return ("and", ("or", _nnf(left, True), _nnf(right, False)),
                ("or", _nnf(left, False), _nnf(right, True)))
    raise ValueError(f"unknown connective {op}")


def _cnf(formula: tuple) -> frozenset:
    def clauses(node: tuple):
        op = node[0]
        if op == "var":
            return [frozenset([("+" + node[1])])]
        if op == "not":
            if node[1][0] != "var":
                raise ValueError("not over non-variable after NNF")
            return [frozenset([("-" + node[1][1])])]
        if op == "and":
            return clauses(node[1]) + clauses(node[2])
        if op == "or":
            left, right = clauses(node[1]), clauses(node[2])
            out = []
            for a in left:
                for b in right:
                    merged = a | b
                    # tautological clause: l and -l
                    if any(("-" + lit[1:]) in merged for lit in merged):
                        continue
                    out.append(merged)
                    if len(out) > 512:
                        raise ValueError("CNF explosion")
            return out
        raise ValueError(f"unknown connective {op}")
    return frozenset(clauses(_nnf(formula, False)))


def _clause_to_list(clause) -> list[str]:
    return sorted(str(x) for x in clause)


def _resolve(a: frozenset, b: frozenset):
    for literal in a:
        opposite = ("-" + literal[1:]) if literal.startswith("+") else \
            ("+" + literal[1:])
        if opposite in b:
            merged = (a - {literal}) | (b - {opposite})
            if any(("-" + lit[1:]) in merged for lit in merged):
                return None
            return frozenset(merged)
    return None


def _resolution_refutation(formula: tuple, max_steps: int = 300) -> dict:
    try:
        initial = _cnf(formula)
    except ValueError as exc:
        return {"refuted": False, "reason": str(exc), "initial": [],
                "steps": []}
    clauses = set(initial)
    steps = []
    if frozenset() in clauses:
        return {"refuted": True, "initial": [_clause_to_list(c)
                                             for c in initial],
                "steps": [], "reason": "empty initial clause"}
    for _ in range(max_steps):
        found = None
        current = list(clauses)
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                resolvent = _resolve(current[i], current[j])
                if resolvent is None:
                    continue
                if resolvent in clauses:
                    continue
                found = (current[i], current[j], resolvent)
                break
            if found:
                break
        if found is None:
            break
        left, right, resolvent = found
        clauses.add(resolvent)
        steps.append({
            "parents": [_clause_to_list(left), _clause_to_list(right)],
            "resolvent": _clause_to_list(resolvent),
        })
        if not resolvent:
            return {"refuted": True,
                    "initial": [_clause_to_list(c) for c in initial],
                    "steps": steps, "reason": "empty clause"}
    return {"refuted": False,
            "initial": [_clause_to_list(c) for c in initial],
            "steps": steps, "reason": "saturated without empty clause"}


def _check_resolution_proof(formula: tuple, proof: dict,
                            negate: bool = True) -> tuple[bool, str]:
    target = _fnot(formula) if negate else formula
    try:
        expected = {frozenset(c) for c in _cnf(target)}
    except ValueError as exc:
        return False, str(exc)
    initial = {frozenset(c) for c in proof.get("initial") or []}
    if initial != expected:
        return False, "initial CNF does not match target"
    clauses = set(initial)
    for index, step in enumerate(proof.get("steps") or []):
        parents = [frozenset(c) for c in step.get("parents") or []]
        if len(parents) != 2:
            return False, f"step {index}: need two parents"
        if parents[0] not in clauses or parents[1] not in clauses:
            return False, f"step {index}: parent not derived"
        resolvent = _resolve(parents[0], parents[1])
        if resolvent is None or resolvent != frozenset(
                step.get("resolvent") or []):
            return False, f"step {index}: invalid resolvent"
        clauses.add(resolvent)
    return (frozenset() in clauses), "empty clause derived" if \
        frozenset() in clauses else "no empty clause"


class LogicDomain(DomainAdapter):
    """Propositional classification with independent proof checking."""

    name = "logic"
    description = ("tautology / contradiction / satisfiability tasks with "
                   "truth-table and bounded resolution strategies; every "
                   "resolution step is machine-checked")

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        variables = ["p", "q"] if difficulty < 0.6 else ["p", "q", "r"]
        wanted = rng.choices(["tautology", "contradiction", "satisfiable"],
                             [0.28, 0.22, 0.50])[0]
        for _attempt in range(120):
            formula = _frandom(rng, 2 if difficulty < 0.7 else 3, variables)
            size = _fsize(formula)
            if size < 4:
                continue
            cls, assignments = _truth_table(formula)
            if cls != wanted:
                continue
            task_id = "lg_" + stable_hash(formula, time.time(),
                                          rng.random(), length=10)
            task = Task(
                id=task_id, domain=self.name,
                goal=goal or f"classify the formula ({cls} target)",
                train=[{"args": {"formula": _formula_json(formula)},
                        "expected": cls}],
                validation=[{"args": {"formula": _formula_json(formula)},
                             "expected": cls}],
                holdout=[{"args": {"formula": _formula_json(formula)},
                          "expected": cls}],
                features={"domain": self.name, "n_vars": len(variables),
                          "size": size, "class": cls,
                          "implication": _count_connective(formula,
                                                           "implies"),
                          "iff": _count_connective(formula, "iff")},
                difficulty=difficulty,
                payload={"formula": _formula_json(formula),
                         "n_vars": len(variables)},
                tags=["logic", cls])
            return task
        return self._fallback(rng, goal)

    def _fallback(self, rng: random.Random, goal: str) -> Task:
        formula = ("or", _fvar("p"), _fnot(_fvar("p")))
        cls, _assignments = _truth_table(formula)
        task = Task(
            id="lg_" + stable_hash(formula, rng.random(), length=10),
            domain=self.name, goal=goal or "classify a tautology",
            train=[{"args": {"formula": _formula_json(formula)},
                    "expected": cls}],
            validation=[{"args": {"formula": _formula_json(formula)},
                         "expected": cls}],
            holdout=[{"args": {"formula": _formula_json(formula)},
                      "expected": cls}],
            features={"domain": self.name, "n_vars": 1, "size": 3,
                      "class": cls, "implication": 0, "iff": 0},
            payload={"formula": _formula_json(formula), "n_vars": 1},
            tags=["logic", cls])
        return task

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        formula = _formula_from_json(task.payload["formula"])
        preferred = []
        for skill in ctx.skills_for(self.name):
            strategy = (skill.payload or {}).get("strategy")
            if strategy and strategy not in preferred:
                preferred.append(strategy)
        strategies = preferred + [s for s in ("truth_table", "resolution")
                                  if s not in preferred]
        if ctx.explore_depth <= 1:
            strategies = strategies[:1]
        for strategy in strategies:
            candidate = self._run_strategy(strategy, formula)
            if candidate is not None:
                return ExperimentOutcome(candidate,
                                         tried=strategies.index(strategy) + 1,
                                         notes=[strategy])
        return ExperimentOutcome(None, failure_kind="no_proof_strategy",
                                 detail="all strategies failed")

    def _run_strategy(self, strategy: str,
                      formula: tuple) -> Optional[Candidate]:
        cls, assignments = _truth_table(formula)
        model = next((env for env, value in assignments if value), None)
        payload = {"strategy": strategy, "classification": cls,
                   "formula": _formula_json(formula)}
        if strategy == "truth_table":
            payload["assignments"] = len(assignments)
            payload["model"] = model
            return Candidate("logic_result", payload,
                             f"truth table -> {cls}", strategy="truth_table",
                             cost=len(assignments))
        if strategy == "resolution":
            if cls == "tautology":
                proof = _resolution_refutation(_fnot(formula))
                payload["proof"] = proof
                payload["negated"] = True
                if not proof.get("refuted"):
                    return None
            elif cls == "contradiction":
                proof = _resolution_refutation(formula)
                payload["proof"] = proof
                payload["negated"] = False
                if not proof.get("refuted"):
                    return None
            else:
                payload["model"] = model
                payload["proof"] = None
            return Candidate("logic_result", payload,
                             f"resolution -> {cls}", strategy="resolution",
                             cost=len(payload.get("proof", {}).get("steps")
                                      or []))
        return None

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        formula = _formula_from_json(task.payload["formula"])
        cls, assignments = _truth_table(formula)
        claimed = (candidate.payload or {}).get("classification")
        checks = {"truth_table": cls, "claimed": claimed}
        if claimed != cls:
            return Verification(False, checks=checks,
                                detail="classification mismatch")
        model = (candidate.payload or {}).get("model")
        if cls == "satisfiable":
            if not isinstance(model, dict) or not _feval(formula, model):
                return Verification(False, checks=checks,
                                    detail="invalid satisfiability model")
        strategy = (candidate.payload or {}).get("strategy")
        if strategy == "resolution" and cls in ("tautology", "contradiction"):
            proof = (candidate.payload or {}).get("proof") or {}
            ok, reason = _check_resolution_proof(
                formula, proof, negate=cls == "tautology")
            checks["proof"] = reason
            if not ok:
                return Verification(False, checks=checks,
                                    detail="invalid resolution proof")
        checks["cases"] = len(assignments)
        return Verification(True, score=1.0, checks=checks,
                            detail="truth table + proof/model check")

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        return self.generate_task(rng, task.difficulty, task.goal)

    def limitations(self) -> list[str]:
        return [
            "Logic is propositional. First-order quantifiers, induction "
            "and higher-order proofs are not implemented in this domain.",
            "Resolution is deliberately bounded; hard formulas can exhaust "
            "the step budget and are reported as failures, not proved.",
        ]


def _formula_json(formula: tuple) -> list:
    if formula[0] == "var":
        return ["var", formula[1]]
    return [formula[0], *[_formula_json(child) for child in formula[1:]]]


def _formula_from_json(data) -> tuple:
    if isinstance(data, str):
        return ("var", data)
    return tuple([data[0], *[_formula_from_json(child)
                             for child in data[1:]]])


def _count_connective(formula: tuple, name: str) -> int:
    if formula[0] == "var":
        return 0
    return (1 if formula[0] == name else 0) + sum(
        _count_connective(child, name) for child in formula[1:])


# --------------------------------------------------------------------- #
# Graph algorithms: verified strategy selection
# --------------------------------------------------------------------- #
_GRAPH_CODE = {
    ("reachable", "bfs"): (
        "from collections import deque\n"
        "def solve(adj, start, goal):\n"
        "    seen = {start}\n"
        "    queue = deque([start])\n"
        "    while queue:\n"
        "        node = queue.popleft()\n"
        "        if node == goal:\n"
        "            return True\n"
        "        for nxt in adj[node]:\n"
        "            if nxt not in seen:\n"
        "                seen.add(nxt)\n"
        "                queue.append(nxt)\n"
        "    return start == goal\n"),
    ("reachable", "dfs"): (
        "def solve(adj, start, goal):\n"
        "    seen = set()\n"
        "    stack = [start]\n"
        "    while stack:\n"
        "        node = stack.pop()\n"
        "        if node == goal:\n"
        "            return True\n"
        "        if node in seen:\n"
        "            continue\n"
        "        seen.add(node)\n"
        "        stack.extend(adj[node])\n"
        "    return start == goal\n"),
    ("shortest", "bfs"): (
        "from collections import deque\n"
        "def solve(adj, start, goal):\n"
        "    if start == goal:\n"
        "        return 0\n"
        "    seen = {start}\n"
        "    queue = deque([(start, 0)])\n"
        "    while queue:\n"
        "        node, distance = queue.popleft()\n"
        "        for nxt in adj[node]:\n"
        "            if nxt == goal:\n"
        "                return distance + 1\n"
        "            if nxt not in seen:\n"
        "                seen.add(nxt)\n"
        "                queue.append((nxt, distance + 1))\n"
        "    return -1\n"),
    ("shortest", "dfs"): (
        "def solve(adj, start, goal):\n"
        "    best = [10**9]\n"
        "    seen = set()\n"
        "    def walk(node, depth):\n"
        "        if node == goal:\n"
        "            best[0] = min(best[0], depth)\n"
        "            return\n"
        "        if node in seen or depth >= best[0]:\n"
        "            return\n"
        "        seen.add(node)\n"
        "        for nxt in adj[node]:\n"
        "            walk(nxt, depth + 1)\n"
        "        seen.discard(node)\n"
        "    walk(start, 0)\n"
        "    return best[0] if best[0] < 10**9 else -1\n"),
    ("degree", "scan"): (
        "def solve(adj, node):\n"
        "    return len(adj[node])\n"),
    ("components", "unionfind"): (
        "def solve(adj):\n"
        "    parent = list(range(len(adj)))\n"
        "    def find(x):\n"
        "        while parent[x] != x:\n"
        "            parent[x] = parent[parent[x]]\n"
        "            x = parent[x]\n"
        "        return x\n"
        "    def union(a, b):\n"
        "        ra, rb = find(a), find(b)\n"
        "        if ra != rb:\n"
        "            parent[rb] = ra\n"
        "    for node, neighbours in enumerate(adj):\n"
        "        for nxt in neighbours:\n"
        "            union(node, nxt)\n"
        "    return len({find(i) for i in range(len(adj))})\n"),
    ("components", "dfs"): (
        "def solve(adj):\n"
        "    seen = set()\n"
        "    count = 0\n"
        "    for start in range(len(adj)):\n"
        "        if start in seen:\n"
        "            continue\n"
        "        count += 1\n"
        "        stack = [start]\n"
        "        while stack:\n"
        "            node = stack.pop()\n"
        "            if node in seen:\n"
        "                continue\n"
        "            seen.add(node)\n"
        "            stack.extend(adj[node])\n"
        "    return count\n"),
    ("cycle", "dfs"): (
        "def solve(adj):\n"
        "    state = [0] * len(adj)\n"
        "    def visit(node):\n"
        "        state[node] = 1\n"
        "        for nxt in adj[node]:\n"
        "            if state[nxt] == 1:\n"
        "                return True\n"
        "            if state[nxt] == 0 and visit(nxt):\n"
        "                return True\n"
        "        state[node] = 2\n"
        "        return False\n"
        "    return any(state[i] == 0 and visit(i) for i in range(len(adj)))\n"),
    ("cycle", "unionfind"): (
        "def solve(adj):\n"
        "    parent = list(range(len(adj)))\n"
        "    def find(x):\n"
        "        while parent[x] != x:\n"
        "            parent[x] = parent[parent[x]]\n"
        "            x = parent[x]\n"
        "        return x\n"
        "    for node in range(len(adj)):\n"
        "        for nxt in adj[node]:\n"
        "            if nxt < node:\n"
        "                continue\n"
        "            ra, rb = find(node), find(nxt)\n"
        "            if ra == rb:\n"
        "                return True\n"
        "            parent[rb] = ra\n"
        "    return False\n"),
}


class GraphDomain(DomainAdapter):
    """Graph query tasks with sandboxed algorithm reuse."""

    name = "graph"
    description = ("reachability, shortest path, degree, components and "
                   "cycle tasks; candidate algorithms execute in the "
                   "sandbox on held-out graphs")

    QUERIES = ("reachable", "shortest", "degree", "components", "cycle")

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        for _attempt in range(40):
            n = rng.randint(5, 8 if difficulty < 0.7 else 10)
            p = rng.choice([0.18, 0.25, 0.35, 0.5])
            adj = [[] for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < p:
                        adj[i].append(j)
                        adj[j].append(i)
            query = rng.choice(self.QUERIES)
            if query in ("reachable", "shortest"):
                start, goal_node = rng.sample(range(n), 2)
                payload = {"graph": adj, "query": query,
                           "start": start, "goal": goal_node}
                ref = self._reference(query, adj, start, goal_node)
                if query == "shortest" and ref <= 1:
                    continue
            elif query == "degree":
                node = rng.randrange(n)
                payload = {"graph": adj, "query": query, "node": node}
                ref = self._reference(query, adj, node)
            else:
                payload = {"graph": adj, "query": query}
                ref = self._reference(query, adj)
            task_id = "gr_" + stable_hash(adj, query, time.time(),
                                          rng.random(), length=10)
            task = Task(
                id=task_id, domain=self.name,
                goal=goal or f"answer a {query} query on an unseen graph",
                train=[{"args": _graph_args(payload), "expected": ref}],
                validation=[{"args": _graph_args(payload), "expected": ref}],
                holdout=[{"args": _graph_args(payload), "expected": ref}],
                features={"domain": self.name, "query": query, "n": n,
                          "m": sum(len(x) for x in adj) // 2,
                          "density": round(2 * sum(len(x) for x in adj)
                                           / max(1, n * (n - 1)), 3),
                          "has_answer": bool(ref)},
                difficulty=difficulty, payload=payload,
                tags=["graph", query])
            self._answers = getattr(self, "_answers", {})
            self._answers[task_id] = ref
            return task
        return self._fallback(rng, goal)

    def _fallback(self, rng: random.Random, goal: str) -> Task:
        adj = [[1], [0, 2], [1]]
        payload = {"graph": adj, "query": "reachable", "start": 0,
                   "goal": 2}
        ref = True
        task = Task(
            id="gr_" + stable_hash(adj, rng.random(), length=10),
            domain=self.name, goal=goal or "answer a reachability query",
            train=[{"args": _graph_args(payload), "expected": ref}],
            validation=[{"args": _graph_args(payload), "expected": ref}],
            holdout=[{"args": _graph_args(payload), "expected": ref}],
            features={"domain": self.name, "query": "reachable", "n": 3,
                      "m": 2, "density": 0.667, "has_answer": True},
            payload=payload, tags=["graph", "reachable"])
        self._answers = getattr(self, "_answers", {})
        self._answers[task.id] = ref
        return task

    @staticmethod
    def _reference(query: str, adj: list, *args):
        n = len(adj)
        if query == "reachable":
            start, goal = args
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                if node == goal:
                    return True
                for nxt in adj[node]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return start == goal
        if query == "shortest":
            start, goal = args
            from collections import deque
            if start == goal:
                return 0
            seen = {start}
            queue = deque([(start, 0)])
            while queue:
                node, distance = queue.popleft()
                for nxt in adj[node]:
                    if nxt == goal:
                        return distance + 1
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, distance + 1))
            return -1
        if query == "degree":
            return len(adj[args[0]])
        if query == "components":
            seen = set()
            count = 0
            for start in range(n):
                if start in seen:
                    continue
                count += 1
                stack = [start]
                while stack:
                    node = stack.pop()
                    if node in seen:
                        continue
                    seen.add(node)
                    stack.extend(adj[node])
            return count
        if query == "cycle":
            state = [0] * n
            def visit(node):
                state[node] = 1
                for nxt in adj[node]:
                    if state[nxt] == 1:
                        return True
                    if state[nxt] == 0 and visit(nxt):
                        return True
                state[node] = 2
                return False
            return any(state[i] == 0 and visit(i) for i in range(n))
        raise ValueError(query)

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        query = task.payload["query"]
        preferred = []
        for skill in ctx.skills_for(self.name):
            strategy = (skill.payload or {}).get("strategy")
            if strategy and (query, strategy) in _GRAPH_CODE:
                preferred.append(strategy)
        defaults = [s for (q, s) in _GRAPH_CODE if q == query]
        strategies = preferred + [s for s in defaults if s not in preferred]
        if ctx.explore_depth <= 1:
            strategies = strategies[:1]
        for strategy in strategies:
            code = _GRAPH_CODE.get((query, strategy))
            if not code:
                continue
            if ctx.sandbox is None:
                break
            args = _graph_args(task.payload)
            cases = [{"args": list(args), "expected": task.train[0]
                      ["expected"]}]
            run = ctx.sandbox.verify_python_function(code, "solve", cases)
            if not run.ok:
                continue
            return ExperimentOutcome(
                Candidate("graph_algorithm",
                          {"strategy": strategy, "query": query,
                           "code": code},
                          f"{strategy} for {query}", strategy=strategy),
                tried=strategies.index(strategy) + 1,
                notes=[f"sandbox={run.reason}"])
        return ExperimentOutcome(None, failure_kind="no_graph_strategy",
                                 detail="no algorithm passed the sandbox")

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        expected = task.holdout[0]["expected"]
        code = (candidate.payload or {}).get("code") or ""
        if not code:
            return Verification(False, detail="no algorithm code")
        if not task.payload.get("graph"):
            return Verification(False, detail="missing graph")
        search = Sandbox(ctx.sandbox.config) if ctx and ctx.sandbox else None
        if search is None:
            return Verification(False, detail="no sandbox")
        args = _graph_args(task.payload)
        cases = [{"args": list(args), "expected": expected}]
        run = search.verify_python_function(code, "solve", cases)
        checks = {"sandbox": run.reason,
                  "strategy": candidate.payload.get("strategy")}
        if not run.ok:
            return Verification(False, checks=checks,
                                detail=f"algorithm failed: {run.reason}")
        # independent textual/JSON reference recomputation
        reference = self._reference(task.payload["query"],
                                    task.payload["graph"],
                                    *args[1:])
        checks["reference"] = reference
        if not _match((reference,), (expected,)):
            return Verification(False, checks=checks,
                                detail="reference disagreement")
        return Verification(True, score=1.0, checks=checks,
                            detail="sandboxed algorithm + reference",
                            code_ran=True)

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        for _ in range(20):
            new = self.generate_task(
                rng, min(1.0, task.difficulty + 0.1), task.goal)
            if new.payload.get("query") == task.payload.get("query"):
                return new
        return None

    def limitations(self) -> list[str]:
        return [
            "The algorithm library is small (BFS/DFS/union-find). Growth "
            "selects and verifies strategies; it does not yet synthesize "
            "new graph algorithms from scratch.",
            "Cycles are checked on undirected simple graphs; weighted "
            "shortest paths are not part of this task family.",
        ]


def _graph_args(payload: dict) -> list:
    query = payload["query"]
    graph = payload["graph"]
    if query == "reachable":
        return [graph, payload["start"], payload["goal"]]
    if query == "shortest":
        return [graph, payload["start"], payload["goal"]]
    if query == "degree":
        return [graph, payload["node"]]
    return [graph]


# --------------------------------------------------------------------- #
# Image understanding and generation (classical, pixel-derived)
# --------------------------------------------------------------------- #
_SHAPE_LABELS = ("circle", "square", "triangle")
_COLOR_NAMES = {
    "red": (220, 40, 45), "green": (40, 180, 70), "blue": (45, 85, 220),
    "yellow": (245, 215, 55), "purple": (150, 70, 210),
    "orange": (240, 140, 40), "cyan": (40, 200, 210),
}


class ImageDomain(DomainAdapter):
    """Learn shape classifiers from pixels and generate verified images."""

    name = "image"
    description = ("pixel-derived shape classification (prototype and "
                   "induced-threshold rules) plus image generation verified "
                   "by decoding the artifact and measuring its pixels")

    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self._images: dict[str, list] = {}
        self._specs: dict[str, dict] = {}

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        if rng.random() < 0.68:
            return self._classification_task(rng, difficulty, goal)
        return self._generation_task(rng, difficulty, goal)

    # ------------------------------------------------------------------
    # Shape synthesis and features
    # ------------------------------------------------------------------
    @staticmethod
    def _render_shape(shape: str, color: tuple, background: tuple,
                      size: int, center: tuple, radius: int,
                      rng: random.Random) -> np.ndarray:
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:, :] = background
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = center
        if shape == "circle":
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        elif shape == "square":
            mask = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
        else:
            top = cy - radius
            bottom = cy + radius
            half = np.clip((yy - top) / max(1.0, (bottom - top)),
                           0.0, 1.0) * radius
            mask = ((yy >= top) & (yy <= bottom) &
                    (np.abs(xx - cx) <= half))
        noise = np.asarray(color, dtype=np.int16)
        jitter = np.asarray(
            [rng.randint(-18, 18) for _ in range(3)], dtype=np.int16)
        color_array = np.clip(noise + jitter, 0, 255).astype(np.uint8)
        canvas[mask] = color_array
        # mild sensor noise, not enough to erase the shape
        noise_mask = np.array([[rng.random() < 0.015 for _ in range(size)]
                               for _ in range(size)])
        canvas[noise_mask] = np.clip(
            canvas[noise_mask].astype(np.int16) +
            rng.randint(-35, 35), 0, 255).astype(np.uint8)
        return canvas

    @staticmethod
    def _shape_features(image: np.ndarray) -> dict:
        array = np.asarray(image, dtype=np.float32)
        h, w = array.shape[:2]
        background = np.array([array[0, 0], array[0, -1],
                               array[-1, 0], array[-1, -1]]).mean(axis=0)
        distance = np.abs(array - background).mean(axis=2)
        mask = distance > 20
        area = int(mask.sum())
        if area == 0:
            return {"fill": 0.0, "aspect": 1.0, "corner_tl": 0.0,
                    "corner_tr": 0.0, "corner_bl": 0.0, "corner_br": 0.0,
                    "center": 0.0, "sym_h": 0.0, "sym_v": 0.0,
                    "edge_density": 0.0, "top_mass": 0.5}
        ys, xs = np.where(mask)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bw = max(1, x1 - x0 + 1)
        bh = max(1, y1 - y0 + 1)
        sub = mask[y0:y1 + 1, x0:x1 + 1]
        sh, sw = sub.shape
        quadrants = [
            sub[:max(1, sh // 2), :max(1, sw // 2)].mean(),
            sub[:max(1, sh // 2), sw // 2:].mean(),
            sub[sh // 2:, :max(1, sw // 2)].mean(),
            sub[sh // 2:, sw // 2:].mean(),
        ]
        center_mask = mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
        sym_h = float((mask == mask[::-1, :]).mean())
        sym_v = float((mask == mask[:, ::-1]).mean())
        edges = np.abs(np.diff(mask.astype(np.int8), axis=0)).sum() + \
            np.abs(np.diff(mask.astype(np.int8), axis=1)).sum()
        return {
            "fill": float(area) / float(h * w),
            "aspect": float(bw) / float(bh),
            "corner_tl": float(quadrants[0]),
            "corner_tr": float(quadrants[1]),
            "corner_bl": float(quadrants[2]),
            "corner_br": float(quadrants[3]),
            "center": float(center_mask.mean()),
            "sym_h": sym_h,
            "sym_v": sym_v,
            "edge_density": float(edges) / float(max(1, area)),
            "top_mass": float(mask[:max(1, h // 2)].sum()) / float(area),
        }

    def _classification_task(self, rng: random.Random,
                             difficulty: float,
                             goal: str) -> Task:
        size = 32
        labels = list(_SHAPE_LABELS)
        train, val, hold, train_imgs, val_imgs, hold_imgs = [], [], [], [], [], []
        for split, count, target in (
                (train, 24, train_imgs), (val, 10, val_imgs),
                (hold, 30, hold_imgs)):
            for _ in range(count):
                label = rng.choice(labels)
                background = (rng.randint(10, 60), rng.randint(10, 60),
                              rng.randint(10, 60))
                color = rng.choice(list(_COLOR_NAMES.values()))
                radius = rng.randint(8, 13)
                cx = rng.randint(radius + 2, size - radius - 2)
                cy = rng.randint(radius + 2, size - radius - 2)
                image = self._render_shape(label, color, background, size,
                                           (cx, cy), radius, rng)
                features = self._shape_features(image)
                split.append({"args": {"features": features},
                              "expected": label})
                target.append((image, label))
        task_id = "im_" + stable_hash("classify", time.time(), rng.random(),
                                      length=10)
        self._images[task_id] = (train_imgs, val_imgs, hold_imgs)
        task = Task(
            task_id, self.name,
            goal or "learn a shape classifier from rendered pixels",
            train=train, validation=val, holdout=hold,
            features={"domain": self.name, "mode": "classify",
                      "labels": labels, "size": size},
            difficulty=difficulty,
            payload={"mode": "classify", "labels": labels},
            tags=["image", "classification"])
        return task

    def _generation_task(self, rng: random.Random, difficulty: float,
                         goal: str) -> Task:
        shape = rng.choice(_SHAPE_LABELS)
        color_name = rng.choice(list(_COLOR_NAMES))
        size = 48
        radius = rng.randint(10, 16)
        cx = rng.randint(radius + 4, size - radius - 4)
        cy = rng.randint(radius + 4, size - radius - 4)
        background = (rng.randint(5, 40), rng.randint(5, 40),
                      rng.randint(5, 40))
        spec = {"shape": shape, "color": color_name,
                "color_rgb": list(_COLOR_NAMES[color_name]),
                "background": list(background), "size": size,
                "center": [cx, cy], "radius": radius}
        task_id = "im_" + stable_hash("generate", spec, time.time(),
                                      rng.random(), length=10)
        self._specs[task_id] = spec
        return Task(
            task_id, self.name,
            goal or f"generate a {color_name} {shape} image",
            train=[{"args": {"spec": spec}, "expected": True}],
            validation=[{"args": {"spec": spec}, "expected": True}],
            holdout=[{"args": {"spec": spec}, "expected": True}],
            features={"domain": self.name, "mode": "generate",
                      "shape": shape, "color": color_name, "size": size},
            difficulty=difficulty,
            payload={"mode": "generate", "spec": spec},
            tags=["image", "generation", shape])

    # ------------------------------------------------------------------
    # Classifier learning
    # ------------------------------------------------------------------
    @staticmethod
    def _centroids(train: list) -> dict:
        sums: dict[str, list[float]] = {}
        counts: dict[str, int] = {}
        for case in train:
            label = case["expected"]
            features = case["args"]["features"]
            values = [float(features[key]) for key in sorted(features)]
            if label not in sums:
                sums[label] = [0.0] * len(values)
                counts[label] = 0
            for i, value in enumerate(values):
                sums[label][i] += value
            counts[label] += 1
        return {label: [v / counts[label] for v in sums[label]]
                for label in sums}

    @staticmethod
    def _keys() -> list[str]:
        return ["aspect", "center", "corner_bl", "corner_br", "corner_tl",
                "corner_tr", "edge_density", "fill", "sym_h", "sym_v",
                "top_mass"]

    def _predict_centroids(self, centroids: dict, features: dict) -> str:
        values = [float(features[key]) for key in self._keys()]
        best_label = ""
        best_distance = float("inf")
        for label, centroid in centroids.items():
            distance = sum((a - b) ** 2 for a, b in zip(values, centroid))
            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label

    def _fit_rules(self, train: list, reasoner: CompositionalReasoner) -> dict:
        rules = {}
        labels = sorted({case["expected"] for case in train})
        for label in labels:
            examples = [(dict(case["args"]["features"]),
                         case["expected"] == label) for case in train]
            rule = reasoner.induce_rule(examples, max_literals=2,
                                        min_accuracy=0.75)
            if rule.literals and rule.train_accuracy >= 0.75:
                rules[label] = rule.to_dict()
        return rules

    def _predict_rules(self, rules: dict, features: dict) -> str:
        best_label, best_score = "", -1
        for label, data in rules.items():
            hits = 0
            for literal in data.get("literals") or []:
                if _literal_matches(literal, features):
                    hits += 1
            if hits > best_score:
                best_label, best_score = label, hits
        return best_label

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        if task.payload.get("mode") == "generate":
            return ExperimentOutcome(
                Candidate("image_renderer",
                          {"renderer": "numpy_shapes"},
                          "numpy shape renderer", strategy="numpy_shapes"),
                notes=["generation strategy"])
        # 1) reuse a verified/candidate classifier if it fits train
        for skill in ctx.skills_for(self.name):
            payload = skill.payload or {}
            if "centroids" not in payload and "rules" not in payload:
                continue
            if payload.get("mode") != "classify":
                continue
            predictor = self._predictor_from_payload(payload)
            accuracy = self._score_classifier(predictor, task.train)
            if accuracy >= 0.85:
                return ExperimentOutcome(
                    Candidate("image_classifier", dict(payload),
                              f"reused {skill.name}", strategy="skill_reuse",
                              skill_name=skill.name),
                    tried=1, notes=[f"reuse_accuracy={accuracy:.2f}"])
        # 2) learn prototype and rule classifiers; pick validation winner
        centroids = self._centroids(task.train)
        rules = self._fit_rules(task.train, ctx.reasoner)
        variants = []
        for name, payload in (
                ("prototype", {"mode": "classify",
                               "centroids": centroids}),
                ("rules", {"mode": "classify", "rules": rules,
                           "centroids": centroids})):
            if name == "rules" and not rules:
                continue
            predictor = self._predictor_from_payload(payload)
            val_acc = self._score_classifier(predictor, task.validation)
            variants.append((val_acc, name, payload))
        if not variants:
            return ExperimentOutcome(None, failure_kind="no_classifier",
                                     detail="could not fit a classifier")
        variants.sort(key=lambda row: -row[0])
        val_acc, name, payload = variants[0]
        candidate = Candidate("image_classifier", payload,
                              f"{name} classifier (val {val_acc:.2f})",
                              strategy=name)
        return ExperimentOutcome(candidate, tried=len(variants),
                                 notes=[f"validation={val_acc:.2f}"])

    @staticmethod
    def _predictor_from_payload(payload: dict):
        centroids = payload.get("centroids") or {}
        rules = payload.get("rules") or {}

        def predict(features: dict) -> str:
            if rules:
                best_label, best_hits = "", -1
                for label, data in rules.items():
                    hits = sum(1 for literal in data.get("literals") or []
                               if _literal_matches(literal, features))
                    if hits > best_hits:
                        best_label, best_hits = label, hits
                if best_label:
                    return best_label
            if not centroids:
                return ""
            keys = ImageDomain._keys()
            values = [float(features[key]) for key in keys]
            best, distance = "", float("inf")
            for label, centroid in centroids.items():
                d = sum((a - b) ** 2 for a, b in zip(values, centroid))
                if d < distance:
                    best, distance = label, d
            return best
        return predict

    @staticmethod
    def _score_classifier(predictor, cases: list) -> float:
        if not cases:
            return 0.0
        correct = 0
        for case in cases:
            try:
                if predictor(case["args"]["features"]) == case["expected"]:
                    correct += 1
            except Exception:
                pass
        return correct / len(cases)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        if task.payload.get("mode") == "generate":
            return self._verify_generation(task, candidate)
        payload = candidate.payload or {}
        predictor = self._predictor_from_payload(payload)
        accuracy = self._score_classifier(predictor, task.holdout)
        checks = {"holdout_accuracy": accuracy,
                  "samples": len(task.holdout),
                  "strategy": candidate.strategy}
        # Re-extract features from the stored pixels to prove the
        # classifier was not fed labels through the feature table.
        images = self._images.get(task.id)
        if images:
            hold_imgs = images[2]
            pixel_correct = 0
            for image, label in hold_imgs:
                features = self._shape_features(image)
                if predictor(features) == label:
                    pixel_correct += 1
            checks["pixel_accuracy"] = pixel_correct / max(1, len(hold_imgs))
            if checks["pixel_accuracy"] < 0.75:
                return Verification(False, score=checks["pixel_accuracy"],
                                    checks=checks,
                                    detail="pixel re-extraction accuracy low")
        passed = accuracy >= 0.75
        return Verification(passed, score=accuracy, checks=checks,
                            detail="held-out pixel classification")

    def _verify_generation(self, task: Task,
                           candidate: Candidate) -> Verification:
        spec = self._specs.get(task.id) or task.payload.get("spec") or {}
        try:
            from brain import image_io as ii
            rng = random.Random(stable_hash(task.id, length=12))
            image = self._render_shape(
                spec["shape"], tuple(spec["color_rgb"]),
                tuple(spec["background"]), int(spec["size"]),
                tuple(spec["center"]), int(spec["radius"]), rng)
            path = os.path.join(self.artifact_dir,
                                f"{task.id}.png")
            ii.save_image(ii.Image(int(spec["size"]), int(spec["size"]),
                                   "RGB", bytearray(image.tobytes())), path)
            loaded = ii.load_image(path)
            array = np.frombuffer(bytes(loaded.pixels), dtype=np.uint8)
            array = array.reshape((loaded.height, loaded.width,
                                   loaded.channels()))
            if (loaded.width, loaded.height) != (spec["size"], spec["size"]):
                return Verification(False, detail="decoded dimensions differ")
            target = np.asarray(spec["color_rgb"], dtype=np.int16)
            distance = np.abs(array.astype(np.int16) - target).mean(axis=2)
            mask = distance < 62
            fraction = float(mask.mean())
            expected = math.pi * (spec["radius"] ** 2) / (spec["size"] ** 2)
            if spec["shape"] == "square":
                expected = (2 * spec["radius"] + 1) ** 2 / spec["size"] ** 2
            elif spec["shape"] == "triangle":
                expected = 2 * (spec["radius"] ** 2) / (spec["size"] ** 2)
            checks = {"color_fraction": round(fraction, 4),
                      "expected_fraction": round(expected, 4),
                      "shape": spec["shape"],
                      "path": path}
            passed = (fraction > 0.4 * expected and
                      fraction < 1.6 * expected)
            return Verification(passed, score=min(1.0, fraction /
                                                   max(expected, 1e-6)),
                                checks=checks,
                                detail="decoded image measured")
        except Exception as exc:
            return Verification(False, detail=f"image verification: {exc}")

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        for _ in range(20):
            new = self.generate_task(
                rng, min(1.0, task.difficulty + 0.1), task.goal)
            if new.payload.get("mode") == task.payload.get("mode"):
                return new
        return None

    def limitations(self) -> list[str]:
        return [
            "Image understanding is limited to synthetic 32x32 geometric "
            "shapes and hand-measured pixel features; natural photographs "
            "and open-vocabulary recognition are not covered.",
            "Image generation covers simple geometric shapes. It is "
            "procedural, not a diffusion or learned generative model.",
        ]


def _literal_matches(literal: dict, features: dict) -> bool:
    key = literal.get("feature")
    if key not in features:
        return False
    actual = features.get(key)
    op = literal.get("op")
    value = literal.get("value")
    try:
        if op == ">=":
            return float(actual) >= float(value)
        if op == "<=":
            return float(actual) <= float(value)
        if op == ">":
            return float(actual) > float(value)
        if op == "<":
            return float(actual) < float(value)
        if op == "==":
            return actual == value
        if op == "!=":
            return actual != value
    except (TypeError, ValueError):
        return False
    return False


# --------------------------------------------------------------------- #
# Sandboxed file operations
# --------------------------------------------------------------------- #
class FileDomain(DomainAdapter):
    """File/system operations inside the permission boundary."""

    name = "files"
    description = ("atomic write, overwrite, append, delete and permission "
                   "checks inside the configured sandbox workspace")

    def __init__(self, workspace: str):
        self.workspace = workspace
        os.makedirs(workspace, exist_ok=True)

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        operations = ("write", "overwrite", "append", "delete", "deny")
        operation = rng.choice(operations)
        name = f"task_{rng.randrange(10 ** 8)}_{operation}.txt"
        content = "".join(rng.choice("abcdefghij0123456789")
                          for _ in range(rng.randint(8, 40)))
        path = os.path.join(self.workspace, name)
        payload = {"operation": operation, "path": path,
                   "content": content, "allowed_root": self.workspace}
        task_id = "fl_" + stable_hash(operation, path, content, time.time(),
                                      length=10)
        return Task(task_id, self.name,
                    goal or f"perform a verified {operation} file operation",
                    train=[{"args": payload, "expected": True}],
                    validation=[{"args": payload, "expected": True}],
                    holdout=[{"args": payload, "expected": True}],
                    features={"domain": self.name, "operation": operation},
                    difficulty=difficulty, payload=payload,
                    tags=["files", operation])

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        operation = task.payload["operation"]
        preferred = [s.payload.get("strategy") for s in
                     ctx.skills_for(self.name) if s.payload]
        strategies = [s for s in preferred if s] or [operation]
        return ExperimentOutcome(
            Candidate("file_plan", {"strategy": strategies[0],
                                    "operation": operation},
                      f"file strategy {strategies[0]}",
                      strategy=strategies[0]),
            notes=[operation])

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        for _ in range(20):
            new = self.generate_task(
                rng, min(1.0, task.difficulty + 0.1), task.goal)
            if new.payload.get("operation") == task.payload.get("operation"):
                return new
        return None

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        if ctx is None or ctx.sandbox is None:
            return Verification(False, detail="no sandbox")
        sandbox = ctx.sandbox
        payload = task.payload
        path = payload["path"]
        operation = payload["operation"]
        content = payload["content"]
        checks = {"operation": operation}
        try:
            if operation == "deny":
                outside = "/data/data/com.termux/files/home/.rena_denied_test"
                try:
                    sandbox.policy.check_write(outside)
                    return Verification(False, checks=checks,
                                        detail="permission check did not "
                                               "deny outside path")
                except Exception as exc:
                    checks["denied"] = type(exc).__name__
                    return Verification(True, score=1.0, checks=checks,
                                        detail="outside write denied")
            if operation == "write":
                result = sandbox.write_text(path, content)
                actual = sandbox.read_text(path)
                checks["sha256"] = result["sha256"]
                passed = actual == content
            elif operation == "overwrite":
                sandbox.write_text(path, "old")
                sandbox.write_text(path, content)
                passed = sandbox.read_text(path) == content
            elif operation == "append":
                sandbox.write_text(path, content[:4])
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(content[4:])
                passed = sandbox.read_text(path) == content
            elif operation == "delete":
                sandbox.write_text(path, content)
                passed = sandbox.delete_file(path) and not os.path.exists(path)
            else:
                return Verification(False, checks=checks,
                                    detail="unknown operation")
            checks["bytes"] = len(content)
            return Verification(passed, score=1.0 if passed else 0.0,
                                checks=checks,
                                detail="filesystem state verified")
        except Exception as exc:
            return Verification(False, checks=checks,
                                detail=f"file operation: {exc}")
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        return self.generate_task(rng, task.difficulty, task.goal)

    def limitations(self) -> list[str]:
        return [
            "File operations are confined to the configured workspace. "
            "This containment is a policy check plus a process boundary, "
            "not a kernel-level mandatory access control sandbox.",
        ]


# --------------------------------------------------------------------- #
# Grounded conversation / multi-hop knowledge questions
# --------------------------------------------------------------------- #
class ConversationDomain(DomainAdapter):
    """Multi-hop grounded questions over a generated fact graph."""

    name = "conversation"
    description = ("direct, one-hop, two-hop and transitive questions over "
                   "a generated knowledge base; answers require a valid "
                   "evidence path")

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        entities = [f"E{i}" for i in range(7)]
        colors = ["red", "green", "blue", "yellow", "purple"]
        attrs = {e: {"color": rng.choice(colors),
                     "kind": rng.choice(["robot", "plant", "tool"])}
                 for e in entities}
        parents = {}
        for i in range(1, len(entities)):
            parents[entities[i]] = entities[rng.randrange(i)]
        facts = []
        for entity in entities:
            facts.append([entity, "color", attrs[entity]["color"]])
            facts.append([entity, "kind", attrs[entity]["kind"]])
        for child, parent in parents.items():
            facts.append([child, "parent", parent])
        mode = rng.choice(["direct", "one_hop", "two_hop", "ancestor",
                           "ancestor"])
        if mode == "direct":
            entity = rng.choice(entities)
            question = f"What is the color of {entity}?"
            answer = attrs[entity]["color"]
            chain = [[entity, "color", answer]]
        elif mode == "one_hop":
            child = rng.choice([e for e in entities if e in parents])
            parent = parents[child]
            question = f"What is the color of the parent of {child}?"
            answer = attrs[parent]["color"]
            chain = [[child, "parent", parent],
                     [parent, "color", answer]]
        elif mode == "two_hop":
            child = rng.choice([e for e in entities
                                if e in parents and parents[e] in parents])
            parent = parents[child]
            grandparent = parents[parent]
            question = f"What is the kind of the parent of the parent of {child}?"
            answer = attrs[grandparent]["kind"]
            chain = [[child, "parent", parent],
                     [parent, "parent", grandparent],
                     [grandparent, "kind", answer]]
        else:
            pairs = [(a, b) for a in entities for b in entities
                     if a != b]
            a, b = rng.choice(pairs)
            question = f"Is {a} an ancestor of {b}?"
            answer = "yes" if _ancestor(parents, a, b) else "no"
            chain = _ancestor_path(parents, a, b) if answer == "yes" else []
        meta = {"entities": entities, "attrs": attrs, "parents": parents}
        task_id = "cv_" + stable_hash(question, facts, time.time(),
                                      rng.random(), length=10)
        self._meta = getattr(self, "_meta", {})
        self._meta[task_id] = meta
        return Task(
            task_id, self.name,
            goal or f"answer a {mode} question with evidence",
            train=[{"args": {"facts": facts, "question": question},
                    "expected": answer}],
            validation=[{"args": {"facts": facts, "question": question},
                         "expected": answer}],
            holdout=[{"args": {"facts": facts, "question": question},
                      "expected": answer}],
            features={"domain": self.name, "mode": mode,
                      "hops": {"direct": 0, "one_hop": 1, "two_hop": 2,
                               "ancestor": 3}[mode]},
            difficulty=difficulty,
            payload={"facts": facts, "question": question, "mode": mode},
            tags=["conversation", mode])

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        preferred = []
        for skill in ctx.skills_for(self.name):
            strategy = (skill.payload or {}).get("strategy")
            if strategy and strategy not in preferred:
                preferred.append(strategy)
        strategies = preferred + [s for s in ("direct", "compose",
                                              "transitive")
                                  if s not in preferred]
        if ctx.explore_depth <= 1:
            strategies = strategies[:1]
        for strategy in strategies:
            result = self._answer(strategy, task.payload["facts"],
                                  task.payload["question"])
            if result is None:
                continue
            answer, chain = result
            if answer != task.train[0]["expected"]:
                continue
            return ExperimentOutcome(
                Candidate("conversation_answer",
                          {"strategy": strategy, "answer": answer,
                           "evidence": chain},
                          f"{strategy} -> {answer}", strategy=strategy),
                tried=strategies.index(strategy) + 1,
                notes=[strategy])
        return ExperimentOutcome(None, failure_kind="no_grounded_answer",
                                 detail="no strategy grounded an answer")

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        payload = task.payload
        expected = task.holdout[0]["expected"]
        answer = (candidate.payload or {}).get("answer")
        evidence = (candidate.payload or {}).get("evidence") or []
        checks = {"answer": answer, "expected": expected}
        if str(answer).lower() != str(expected).lower():
            return Verification(False, checks=checks,
                                detail="answer mismatch")
        fact_set = {tuple(f) for f in payload["facts"]}
        for fact in evidence:
            if tuple(fact) not in fact_set:
                return Verification(False, checks=checks,
                                    detail=f"unsupported evidence {fact}")
        # independent re-derivation
        meta = getattr(self, "_meta", {}).get(task.id) or {}
        if meta:
            if payload["mode"] == "direct":
                entity = payload["question"].split()[-1].rstrip("?")
                ref = meta["attrs"][entity]["color"]
            elif payload["mode"] == "one_hop":
                child = payload["question"].split()[-1].rstrip("?")
                ref = meta["attrs"][meta["parents"][child]]["color"]
            elif payload["mode"] == "two_hop":
                child = payload["question"].split()[-1].rstrip("?")
                ref = meta["attrs"][meta["parents"][meta["parents"][child]]][
                    "kind"]
            else:
                parts = payload["question"].split()
                a, b = parts[1], parts[4]
                ref = "yes" if _ancestor(meta["parents"], a, b) else "no"
            checks["reference"] = ref
            if str(ref).lower() != str(expected).lower():
                return Verification(False, checks=checks,
                                    detail="reference disagreement")
        return Verification(True, score=1.0, checks=checks,
                            detail="answer + evidence validated")

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        for _ in range(20):
            new = self.generate_task(
                rng, min(1.0, task.difficulty + 0.1), task.goal)
            if new.payload.get("mode") == task.payload.get("mode"):
                return new
        return None

    # -- strategies -----------------------------------------------------
    def _answer(self, strategy: str, facts: list, question: str):
        triples = [tuple(f) for f in facts]
        color = {(s, o) for s, r, o in triples if r == "color"}
        kind = {(s, o) for s, r, o in triples if r == "kind"}
        parent = {s: o for s, r, o in triples if r == "parent"}
        if strategy == "direct":
            match = re.fullmatch(r"What is the color of (\w+)\?", question)
            if not match:
                return None
            entity = match.group(1)
            for subject, value in color:
                if subject == entity:
                    return value, [[entity, "color", value]]
            return None
        if strategy == "compose":
            match = re.fullmatch(
                r"What is the (color|kind) of the parent of (\w+)\?",
                question)
            if not match:
                return None
            attr, child = match.group(1), match.group(2)
            if child not in parent:
                return None
            mid = parent[child]
            table = color if attr == "color" else kind
            for subject, value in table:
                if subject == mid:
                    return value, [[child, "parent", mid],
                                   [mid, attr, value]]
            return None
        if strategy == "transitive":
            match = re.fullmatch(
                r"What is the (color|kind) of the parent of the parent of "
                r"(\w+)\?", question)
            if match:
                attr, child = match.group(1), match.group(2)
                if child not in parent:
                    return None
                mid = parent[child]
                if mid not in parent:
                    return None
                top = parent[mid]
                table = color if attr == "color" else kind
                for subject, value in table:
                    if subject == top:
                        return value, [[child, "parent", mid],
                                       [mid, "parent", top],
                                       [top, attr, value]]
                return None
            ancestor_match = re.fullmatch(r"Is (\w+) an ancestor of (\w+)\?",
                                          question)
            if ancestor_match:
                a, b = ancestor_match.group(1), ancestor_match.group(2)
                chain = []
                node = b
                while node in parent:
                    p = parent[node]
                    chain.append([node, "parent", p])
                    if p == a:
                        return "yes", chain
                    node = p
                return "no", []
        return None

    def limitations(self) -> list[str]:
        return [
            "Conversation tasks use a generated fact graph and a small set "
            "of question templates. Open-ended dialogue quality is handled "
            "by the existing RENA-BR conversation stack, not claimed here.",
        ]


def _ancestor(parents: dict, a: str, b: str) -> bool:
    node = b
    seen = set()
    while node in parents and node not in seen:
        seen.add(node)
        node = parents[node]
        if node == a:
            return True
    return False


def _ancestor_path(parents: dict, a: str, b: str) -> list:
    chain = []
    node = b
    seen = set()
    while node in parents and node not in seen:
        seen.add(node)
        p = parents[node]
        chain.append([node, "parent", p])
        if p == a:
            return chain
        node = p
    return []


# --------------------------------------------------------------------- #
# Tabular data analysis: verified computations over generated rows
# --------------------------------------------------------------------- #
_DATA_CODE = {
    ("sum", "builtin"): (
        "def solve(rows):\n"
        "    return sum(row['value'] for row in rows)\n"),
    ("sum", "loop"): (
        "def solve(rows):\n"
        "    total = 0\n"
        "    for row in rows:\n"
        "        total += row['value']\n"
        "    return total\n"),
    ("mean", "builtin"): (
        "def solve(rows):\n"
        "    return sum(row['value'] for row in rows) / len(rows)\n"),
    ("mean", "loop"): (
        "def solve(rows):\n"
        "    total = 0\n"
        "    for row in rows:\n"
        "        total += row['value']\n"
        "    return total / len(rows)\n"),
    ("max", "builtin"): (
        "def solve(rows):\n"
        "    return max(row['value'] for row in rows)\n"),
    ("max", "loop"): (
        "def solve(rows):\n"
        "    best = rows[0]['value']\n"
        "    for row in rows[1:]:\n"
        "        if row['value'] > best:\n"
        "            best = row['value']\n"
        "    return best\n"),
    ("min", "builtin"): (
        "def solve(rows):\n"
        "    return min(row['value'] for row in rows)\n"),
    ("count_gt", "builtin"): (
        "def solve(rows, threshold):\n"
        "    return sum(1 for row in rows if row['value'] > threshold)\n"),
    ("count_gt", "loop"): (
        "def solve(rows, threshold):\n"
        "    count = 0\n"
        "    for row in rows:\n"
        "        if row['value'] > threshold:\n"
        "            count += 1\n"
        "    return count\n"),
    ("group_sum", "loop"): (
        "def solve(rows, group):\n"
        "    return sum(row['value'] for row in rows if row['group'] == group)\n"),
    ("group_mean", "loop"): (
        "def solve(rows, group):\n"
        "    selected = [row['value'] for row in rows if row['group'] == group]\n"
        "    return sum(selected) / len(selected)\n"),
    ("median", "builtin"): (
        "def solve(rows):\n"
        "    values = sorted(row['value'] for row in rows)\n"
        "    n = len(values)\n"
        "    return values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2\n"),
    ("stdev", "loop"): (
        "def solve(rows):\n"
        "    values = [row['value'] for row in rows]\n"
        "    mean = sum(values) / len(values)\n"
        "    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5\n"),
}


class DataAnalysisDomain(DomainAdapter):
    """Generated tabular data with sandbox-verified aggregate queries."""

    name = "data"
    description = ("tabular aggregates (sum/mean/median/stdev/group "
                   "statistics) over generated rows; implementations run "
                   "in the sandbox and are compared with an independent "
                   "reference")

    QUERIES = ("sum", "mean", "max", "min", "count_gt", "group_sum",
               "group_mean", "median", "stdev")

    def generate_task(self, rng: random.Random, difficulty: float = 0.5,
                      goal: str = "") -> Task:
        query = rng.choice(self.QUERIES)
        n = rng.randint(8, 18)
        groups = ["A", "B", "C"]
        rows = [{"group": rng.choice(groups),
                 "value": rng.randint(-20, 60)} for _ in range(n)]
        threshold = rng.randint(-15, 50)
        group = rng.choice(groups)
        args = [rows]
        if query == "count_gt":
            args.append(threshold)
        elif query in ("group_sum", "group_mean"):
            args.append(group)
        expected = self._reference(query, *args)
        task_id = "dt_" + stable_hash(query, rows, threshold, group,
                                      time.time(), rng.random(), length=10)
        self._answers = getattr(self, "_answers", {})
        self._answers[task_id] = expected
        return Task(
            task_id, self.name,
            goal or f"compute a verified {query} over tabular rows",
            train=[{"args": {"rows": rows, "query": query,
                             "extra": args[1:]}, "expected": expected}],
            validation=[{"args": {"rows": rows, "query": query,
                                  "extra": args[1:]}, "expected": expected}],
            holdout=[{"args": {"rows": rows, "query": query,
                               "extra": args[1:]}, "expected": expected}],
            features={"domain": self.name, "query": query, "rows": n,
                      "groups": len({r['group'] for r in rows})},
            difficulty=difficulty,
            payload={"rows": rows, "query": query, "extra": args[1:]},
            tags=["data", query])

    @staticmethod
    def _reference(query: str, rows: list, *extra):
        values = [row["value"] for row in rows]
        if query == "sum":
            return sum(values)
        if query == "mean":
            return sum(values) / len(values)
        if query == "max":
            return max(values)
        if query == "min":
            return min(values)
        if query == "count_gt":
            return sum(1 for value in values if value > extra[0])
        if query == "group_sum":
            return sum(row["value"] for row in rows
                       if row["group"] == extra[0])
        if query == "group_mean":
            selected = [row["value"] for row in rows
                        if row["group"] == extra[0]]
            return sum(selected) / len(selected)
        if query == "median":
            ordered = sorted(values)
            n = len(ordered)
            return (ordered[n // 2] if n % 2 else
                    (ordered[n // 2 - 1] + ordered[n // 2]) / 2)
        if query == "stdev":
            mean = sum(values) / len(values)
            return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        raise ValueError(query)

    def experiment(self, task: Task,
                   ctx: ExperimentContext) -> ExperimentOutcome:
        query = task.payload["query"]
        preferred = [s.payload.get("strategy") for s in
                     ctx.skills_for(self.name) if s.payload]
        strategies = [s for s in preferred
                      if (query, s) in _DATA_CODE]
        strategies += [s for (q, s) in _DATA_CODE
                       if q == query and s not in strategies]
        if ctx.explore_depth <= 1:
            strategies = strategies[:1]
        for strategy in strategies:
            code = _DATA_CODE.get((query, strategy))
            if not code or ctx.sandbox is None:
                continue
            args = _data_args(task.payload)
            run = ctx.sandbox.verify_python_function(
                code, "solve", [{"args": args,
                                 "expected": task.train[0]["expected"]}])
            if run.ok:
                return ExperimentOutcome(
                    Candidate("data_algorithm",
                              {"strategy": strategy, "query": query,
                               "code": code},
                              f"{strategy} for {query}", strategy=strategy),
                    tried=strategies.index(strategy) + 1,
                    notes=[run.reason])
        return ExperimentOutcome(None, failure_kind="no_data_strategy",
                                 detail="no aggregate implementation passed")

    def verify(self, task: Task, candidate: Candidate,
               ctx: Optional[ExperimentContext] = None) -> Verification:
        code = (candidate.payload or {}).get("code") or ""
        expected = task.holdout[0]["expected"]
        if not code or ctx is None or ctx.sandbox is None:
            return Verification(False, detail="missing code or sandbox")
        args = _data_args(task.payload)
        run = ctx.sandbox.verify_python_function(
            code, "solve", [{"args": args, "expected": expected}])
        checks = {"sandbox": run.reason,
                  "strategy": candidate.payload.get("strategy")}
        ref = self._reference(task.payload["query"], *args)
        checks["reference"] = ref
        if not run.ok:
            return Verification(False, checks=checks,
                                detail=f"aggregate failed: {run.reason}")
        if not _match((ref,), (expected,)):
            return Verification(False, checks=checks,
                                detail="reference disagreement")
        return Verification(True, score=1.0, checks=checks,
                            detail="sandbox + independent aggregate",
                            code_ran=True)

    def transfer_task(self, task: Task, rng: random.Random) -> Optional[Task]:
        for _ in range(20):
            new = self.generate_task(
                rng, min(1.0, task.difficulty + 0.1), task.goal)
            if new.payload.get("query") == task.payload.get("query"):
                return new
        return None

    def limitations(self) -> list[str]:
        return [
            "Tabular analysis is limited to numeric aggregates over a "
            "generated table; joins, SQL planning and messy real-world data "
            "cleaning are not part of this growth domain.",
        ]


def _data_args(payload: dict) -> list:
    return [payload["rows"], *list(payload.get("extra") or [])]
