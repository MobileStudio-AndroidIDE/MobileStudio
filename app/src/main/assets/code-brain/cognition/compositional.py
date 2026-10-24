"""Compositional and relational reasoning built on VSA + rules.

Capabilities here are structural, not lookup tables:

* role/filler unbinding and cleanup,
* analogical mapping ``A:B :: C:?``,
* rule induction from labelled examples (conjunctive literals),
* transitive closure over relation pairs,
* compositional sequence decoding.

Every induced rule carries its supporting literals and measured
accuracy so downstream promotion can verify it before registration.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np

from cognition.vsa import VSA


@dataclass
class Literal:
    feature: str
    op: str
    value: object

    def matches(self, example: dict) -> bool:
        if self.feature not in example:
            return False
        actual = example.get(self.feature)
        try:
            if self.op == "==":
                return actual == self.value
            if self.op == "!=":
                return actual != self.value
            if self.op == ">=":
                return float(actual) >= float(self.value)
            if self.op == "<=":
                return float(actual) <= float(self.value)
            if self.op == ">":
                return float(actual) > float(self.value)
            if self.op == "<":
                return float(actual) < float(self.value)
            if self.op == "in":
                return actual in self.value
            if self.op == "not_in":
                return actual not in self.value
        except (TypeError, ValueError):
            return False
        return False

    def to_dict(self) -> dict:
        return {"feature": self.feature, "op": self.op,
                "value": self.value}

    def describe(self) -> str:
        return f"{self.feature} {self.op} {self.value!r}"


@dataclass
class InducedRule:
    literals: list[Literal]
    default: bool = False
    positives: int = 0
    negatives: int = 0
    train_accuracy: float = 0.0
    holdout_accuracy: float = 0.0
    description: str = ""

    def predict(self, example: dict) -> bool:
        return all(lit.matches(example) for lit in self.literals)

    def to_dict(self) -> dict:
        return {
            "literals": [lit.to_dict() for lit in self.literals],
            "default": self.default,
            "positives": self.positives,
            "negatives": self.negatives,
            "train_accuracy": round(self.train_accuracy, 4),
            "holdout_accuracy": round(self.holdout_accuracy, 4),
            "description": self.description,
        }


class CompositionalReasoner:
    def __init__(self, vsa: VSA):
        self.vsa = vsa
        self.known_symbols: set[str] = set()

    # ------------------------------------------------------------------
    # VSA queries
    # ------------------------------------------------------------------
    def remember_symbol(self, name: str) -> None:
        self.known_symbols.add(str(name))
        self.vsa.register(str(name))

    def query(self, composite, role: str,
              k: int = 1) -> list[tuple[str, float]]:
        probe = self.vsa.unbind(composite, self.vsa.symbol(role))
        return self.vsa.cleanup(probe, k=k)

    def query_many(self, composite, roles: Iterable[str]) -> dict:
        return {role: self.query(composite, role, k=1)
                for role in roles}

    def analogy(self, a: str, b: str, c: str,
                k: int = 1) -> list[tuple[str, float]]:
        """Solve ``A:B :: C:?`` with role-filler algebra."""
        relation = self.vsa.bind(self.vsa.symbol(a), self.vsa.symbol(b))
        probe = self.vsa.unbind(relation, self.vsa.symbol(c))
        return self.vsa.cleanup(probe, k=k)

    def compose(self, pairs: Iterable[tuple]) -> np.ndarray:
        return self.vsa.encode_role_fillers(pairs)

    def sequence_at(self, sequence_vector, position: int,
                    k: int = 1) -> list[tuple[str, float]]:
        probe = self.vsa.invert_permute(sequence_vector, position)
        return self.vsa.cleanup(probe, k=k)

    # ------------------------------------------------------------------
    # Relational closure
    # ------------------------------------------------------------------
    @staticmethod
    def transitive_closure(pairs: Iterable[tuple],
                           max_rounds: int = 8) -> set[tuple]:
        closure = {(str(a), str(b)) for a, b in pairs}
        for _ in range(max(1, int(max_rounds))):
            additions = set()
            by_source: dict[str, set[str]] = {}
            for a, b in closure:
                by_source.setdefault(a, set()).add(b)
            for a, b in closure:
                for c in by_source.get(b, ()):
                    if (a, c) not in closure:
                        additions.add((a, c))
            if not additions:
                break
            closure |= additions
        return closure

    # ------------------------------------------------------------------
    # Rule induction
    # ------------------------------------------------------------------
    def induce_rule(self, examples: Sequence[tuple[dict, bool]],
                    holdout: Optional[Sequence[tuple[dict, bool]]] = None,
                    max_literals: int = 3,
                    min_accuracy: float = 0.7) -> InducedRule:
        data = [(dict(x), bool(y)) for x, y in examples]
        if not data:
            return InducedRule([], True, 0, 0, 1.0, 1.0,
                               "no data -> default")
        candidates = self._candidate_literals(data)
        best: Optional[InducedRule] = None
        for size in range(1, max(1, int(max_literals)) + 1):
            for combo in itertools.combinations(candidates, size):
                literals = list(combo)
                accuracy = self._accuracy(literals, data)
                if accuracy < min_accuracy:
                    continue
                positives = sum(1 for ex, y in data
                                if y and all(l.matches(ex) for l in literals))
                negatives = sum(1 for ex, y in data
                                if (not y) and all(l.matches(ex)
                                                   for l in literals))
                score = accuracy - 0.01 * len(literals)
                if best is None or score > (best.train_accuracy
                                            - 0.01 * len(best.literals)):
                    holdout_accuracy = (self._accuracy(literals, list(holdout))
                                        if holdout else 0.0)
                    best = InducedRule(
                        literals=literals, default=False,
                        positives=positives, negatives=negatives,
                        train_accuracy=accuracy,
                        holdout_accuracy=holdout_accuracy,
                        description=" AND ".join(l.describe()
                                                 for l in literals))
            if best is not None and best.train_accuracy >= 0.999:
                break
        if best is None:
            majority = sum(1 for _x, y in data if y) >= len(data) / 2
            best = InducedRule([], majority, 0, 0,
                               self._accuracy([], data),
                               self._accuracy([], list(holdout))
                               if holdout else 0.0,
                               "majority default")
        # greedily drop literals that do not hurt accuracy
        if len(best.literals) > 1:
            current = list(best.literals)
            for literal in list(current):
                trial = [l for l in current if l is not literal]
                if self._accuracy(trial, data) >= best.train_accuracy - 1e-9:
                    current = trial
            if len(current) < len(best.literals):
                holdout_accuracy = (self._accuracy(current, list(holdout))
                                    if holdout else 0.0)
                best = InducedRule(
                    literals=current, default=False,
                    positives=best.positives, negatives=best.negatives,
                    train_accuracy=self._accuracy(current, data),
                    holdout_accuracy=holdout_accuracy,
                    description=" AND ".join(l.describe()
                                             for l in current) or
                    "majority default")
        return best

    @staticmethod
    def _accuracy(literals: Iterable[Literal],
                  data: Sequence[tuple[dict, bool]]) -> float:
        literals = list(literals)
        if not data:
            return 0.0
        correct = sum(1 for example, label in data
                      if all(lit.matches(example) for lit in literals) == label)
        return correct / len(data)

    @staticmethod
    def _candidate_literals(data: Sequence[tuple[dict, bool]]
                            ) -> list[Literal]:
        values_by_feature: dict[str, set] = {}
        for example, _label in data:
            for key, value in example.items():
                values_by_feature.setdefault(str(key), set())
                if isinstance(value, (int, float, bool)):
                    values_by_feature[str(key)].add(value)
                else:
                    values_by_feature[str(key)].add(str(value))
        literals: list[Literal] = []
        for feature, values in values_by_feature.items():
            clean_values = [v for v in values if not _is_nan(v)]
            if len(clean_values) <= 6:
                for value in sorted(clean_values, key=str):
                    literals.append(Literal(feature, "==", value))
                    literals.append(Literal(feature, "!=", value))
            numeric = [float(v) for v in clean_values
                       if isinstance(v, (int, float))
                       and not isinstance(v, bool)]
            if len(numeric) >= 2:
                unique = sorted(set(numeric))
                thresholds = {(unique[i] + unique[i + 1]) / 2.0
                              for i in range(len(unique) - 1)}
                for threshold in sorted(thresholds)[:6]:
                    literals.append(Literal(feature, ">=", threshold))
                    literals.append(Literal(feature, "<=", threshold))
        # drop literals that classify nothing / everything
        filtered = []
        for literal in literals:
            hits = sum(1 for example, _y in data if literal.matches(example))
            if 0 < hits < len(data):
                filtered.append(literal)
        return filtered[:48]


def _is_nan(value: object) -> bool:
    try:
        return float(value) != float(value)
    except (TypeError, ValueError):
        return False
