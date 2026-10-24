"""Hypothesis Evaluator.

Evaluates and ranks algorithmic hypotheses based on:
  - Correctness: Does the approach solve the problem type?
  - Complexity: Is the time/space complexity sufficient for constraints?
  - Risk: How likely is implementation to fail?
  - Coverage: Does it cover the structural hints?

Selects the best hypothesis. No LLM involved.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

# A tie-break term from learned execution outcomes is applied ONLY when
# the store holds at least this many records (MIN_OUTCOMES_FOR_RANKING
# lives in the learning store); with an empty (or cold) store the
# ranking is exactly what it always was.
_PRIOR_WEIGHT = 0.05


@dataclass
class EvaluationCriteria:
    """Criteria for evaluating a hypothesis."""
    correctness_weight: float = 0.4
    complexity_weight: float = 0.3
    risk_weight: float = 0.2
    coverage_weight: float = 0.1


@dataclass
class EvaluationResult:
    """Result of evaluating a single hypothesis."""
    hypothesis_id: int
    hypothesis_name: str
    correctness_score: float  # 0.0 - 1.0
    complexity_score: float  # 0.0 - 1.0
    risk_score: float  # 0.0 - 1.0 (lower is better, but we normalize to higher = better)
    coverage_score: float  # 0.0 - 1.0
    total_score: float  # weighted sum
    issues: list[str] = field(default_factory=list)


class HypothesisEvaluator:
    """Evaluates and selects the best algorithmic hypothesis.

    Evaluation dimensions:
      1. Correctness: Does the approach match the problem type?
      2. Complexity: Is it efficient enough for the constraints?
      3. Risk: How likely is the implementation to be correct?
      4. Coverage: Does it cover the structural hints?
    """

    def __init__(self, criteria: Optional[EvaluationCriteria] = None):
        self.criteria = criteria or EvaluationCriteria()

    def evaluate_all(
        self,
        hypotheses: list,
        structured_problem,
    ) -> list[EvaluationResult]:
        """Evaluate all hypotheses and return ranked results."""
        results = []
        for h in hypotheses:
            result = self.evaluate(h, structured_problem)
            results.append(result)

        # CONSUME (learning): a SMALL tie-break term from the learned
        # execution-outcome priors.  No-op with an empty/cold store
        # (< MIN_OUTCOMES_FOR_RANKING records) — see the guard below.
        prior_bonus = self._strategy_prior_bonus(hypotheses,
                                                 structured_problem)
        for result in results:
            bonus = prior_bonus.get(result.hypothesis_name, 0.0)
            if bonus:
                result.total_score += bonus

        results.sort(key=lambda r: r.total_score, reverse=True)
        return results

    @staticmethod
    def _strategy_prior_bonus(hypotheses: list,
                              structured_problem) -> dict:
        """{hypothesis_name: bonus} from learned priors, or {} when cold."""
        try:
            from learning.learner.code_learner import (
                LearningFromOutcome, MIN_OUTCOMES_FOR_RANKING,
            )
            store = LearningFromOutcome()
            if store.total_outcomes() < MIN_OUTCOMES_FOR_RANKING:
                return {}
            tags = [str(t).strip().lower()
                    for t in (getattr(structured_problem,
                                      "structural_hints", None) or [])
                    if str(t).strip()]
            tags += [str(t).strip().lower()
                     for t in (getattr(structured_problem,
                                       "data_types", None) or [])
                     if str(t).strip()]
            shape = "+".join(sorted(set(tags))) or "unknown"
            bonus = {}
            for h in hypotheses:
                name = str(getattr(h, "name", "") or "")
                if not name or name in bonus:
                    continue
                prior = store.strategy_prior(name, shape)
                bonus[name] = _PRIOR_WEIGHT * (prior - 0.5)
            return bonus
        except Exception:
            return {}

    def evaluate(self, hypothesis, structured_problem) -> EvaluationResult:
        """Evaluate a single hypothesis."""
        correctness = self._score_correctness(hypothesis, structured_problem)
        complexity = self._score_complexity(hypothesis, structured_problem)
        risk = self._score_risk(hypothesis, structured_problem)
        coverage = self._score_coverage(hypothesis, structured_problem)

        # Normalize risk: higher risk_score = lower risk (more desirable)
        risk_normalized = 1.0 - risk

        total = (
            self.criteria.correctness_weight * correctness +
            self.criteria.complexity_weight * complexity +
            self.criteria.risk_weight * risk_normalized +
            self.criteria.coverage_weight * coverage
        )

        issues = []
        if correctness < 0.3:
            issues.append("Low correctness: approach may not solve this problem type")
        if complexity < 0.3:
            issues.append("Low complexity score: may be too slow for constraints")
        if risk > 0.7:
            issues.append("High implementation risk")

        return EvaluationResult(
            hypothesis_id=hypothesis.id,
            hypothesis_name=hypothesis.name,
            correctness_score=correctness,
            complexity_score=complexity,
            risk_score=risk_normalized,
            coverage_score=coverage,
            total_score=total,
            issues=issues,
        )

    def _score_correctness(self, h, sp) -> float:
        """Score how likely the hypothesis is to correctly solve the problem."""
        score = 0.5  # baseline

        # Check if the approach matches problem type
        if sp.data_types:
            type_set = set(sp.data_types)

            # Graph problems need graph algorithms
            if "graph" in type_set:
                if any(p in h.algorithmic_primitives for p in ("bfs", "dfs")):
                    score += 0.4
                elif "graph_traversal" in h.name:
                    score += 0.4
                else:
                    score -= 0.2

            # Tree problems need tree algorithms
            if "tree" in type_set:
                if any(p in h.algorithmic_primitives for p in ("bfs", "dfs")):
                    score += 0.3
                else:
                    score -= 0.1

            # Matrix problems
            if "matrix" in type_set:
                if "simulation" in h.algorithmic_primitives:
                    score += 0.2

        # Check structural hints
        hints = set(sp.structural_hints)
        tags = set(h.approach_tags)

        if "pair_search" in hints and "pointer" in tags:
            score += 0.2
        if "contiguous_subarray" in hints and "subarray" in tags:
            score += 0.2
        if "sorting" in hints and "sort" in tags:
            score += 0.1
        if "prefix_sum" in hints and "prefix" in tags:
            score += 0.2

        # Simple heuristic: for N <= 20, any approach is fine
        for c in sp.constraints:
            if c.max_value and c.max_value <= 20:
                score += 0.1
                break

        return max(0.0, min(1.0, score))

    def _score_complexity(self, h, sp) -> float:
        """Score whether the complexity is sufficient for the constraints."""
        # Find the maximum constraint value
        max_n = 1000  # default assumption
        for c in sp.constraints:
            if c.max_value:
                max_n = max(max_n, c.max_value)

        # Parse the time complexity
        tc = h.time_complexity

        # Estimate maximum operations allowed (10^8 for 1 second)
        max_ops = 10 ** 8

        # Compute estimated operations based on complexity
        estimated_ops = self._estimate_operations(tc, max_n)

        if estimated_ops <= max_ops:
            return 1.0
        elif estimated_ops <= max_ops * 10:
            return 0.7
        elif estimated_ops <= max_ops * 100:
            return 0.4
        else:
            return 0.1

    def _estimate_operations(self, complexity: str, n: int) -> float:
        """Estimate the number of operations for a given complexity."""
        if "O(1)" in complexity or "O(log" in complexity:
            return 100
        if "O(n)" in complexity and "O(n^2)" not in complexity and "O(n log" not in complexity:
            return n
        if "O(n log n)" in complexity:
            return n * max(1, n.bit_length())
        if "O(n^2)" in complexity:
            return n * n
        if "O(n^3)" in complexity:
            return n * n * n
        if "O(n*m)" in complexity:
            return n * n  # assume m ~ n
        if "O(V + E)" in complexity:
            return n * 2  # V ~ n, E ~ n
        if "O(2^n)" in complexity:
            return 2 ** min(n, 30)  # cap for safety
        if "O(n!)" in complexity:
            return min(n, 12) ** min(n, 12)  # cap for safety
        return n * n  # default: assume quadratic

    def _score_risk(self, h, sp) -> float:
        """Score implementation risk (0.0 = no risk, 1.0 = high risk)."""
        risk = 0.0

        # Complex algorithms are riskier
        if "dynamic_programming" in h.algorithmic_primitives:
            risk += 0.4
        if "graph_traversal" in h.name:
            risk += 0.3
        if len(h.algorithmic_primitives) > 2:
            risk += 0.2

        # Simple algorithms are less risky
        if "linear_scan" in h.algorithmic_primitives:
            risk -= 0.2
        if "counting" in h.algorithmic_primitives:
            risk -= 0.1
        if "simulation" in h.algorithmic_primitives:
            risk -= 0.1
        if "math_formula" in h.algorithmic_primitives:
            risk -= 0.2

        # Fewer data structures = less risk
        if len(h.data_structures) <= 1:
            risk -= 0.1

        return max(0.0, min(1.0, risk))

    def _score_coverage(self, h, sp) -> float:
        """Score how well the hypothesis covers the problem's structural hints."""
        if not sp.structural_hints:
            return 0.5  # no hints = neutral

        hints = set(sp.structural_hints)
        tags = set(h.approach_tags)

        # Check overlap
        covered = 0
        for hint in hints:
            for tag in tags:
                if hint in tag or tag in hint:
                    covered += 1
                    break

        if not hints:
            return 0.5
        return covered / len(hints)

    def select_best(self, evaluation_results: list[EvaluationResult]):
        """Select the best hypothesis from evaluation results."""
        if not evaluation_results:
            return None
        return evaluation_results[0]
