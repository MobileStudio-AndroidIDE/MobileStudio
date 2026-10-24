"""Hypothesis Generator - produces multiple solution candidates.

Non-determinism comes from exploring multiple valid reasoning paths,
not from random selection. Each hypothesis represents a genuinely
different approach to solving the problem.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from brain.generation.events import Hypothesis, EventType, GenerationEvent


@dataclass
class HypothesisConfig:
    max_hypotheses: int = 3
    min_confidence: float = 0.3
    explore_alternatives: bool = True


# Each operation maps to a list of genuinely different implementation strategies.
# The system chooses from these based on context, not randomly.
_ALGORITHMIC_STRATEGIES: dict[str, list[dict]] = {
    "find_max": [
        {"id": "builtin", "desc": "Use built-in max()", "confidence": 0.9,
         "code_hint": "return max(data)", "approach": "builtin"},
        {"id": "linear_scan", "desc": "Manual linear scan tracking maximum",
         "confidence": 0.85,
         "code_hint": "result = data[0]\nfor x in data[1:]:\n    if x > result: result = x\nreturn result",
         "approach": "linear_scan"},
        {"id": "sort_desc", "desc": "Sort descending and take first element",
         "confidence": 0.7,
         "code_hint": "return sorted(data, reverse=True)[0]",
         "approach": "sort_then_select"},
    ],
    "find_min": [
        {"id": "builtin", "desc": "Use built-in min()", "confidence": 0.9,
         "code_hint": "return min(data)", "approach": "builtin"},
        {"id": "linear_scan", "desc": "Manual linear scan tracking minimum",
         "confidence": 0.85,
         "code_hint": "result = data[0]\nfor x in data[1:]:\n    if x < result: result = x\nreturn result",
         "approach": "linear_scan"},
        {"id": "sort_asc", "desc": "Sort ascending and take first element",
         "confidence": 0.7,
         "code_hint": "return sorted(data)[0]",
         "approach": "sort_then_select"},
    ],
    "sort_descending": [
        {"id": "builtin", "desc": "Use Python sorted(reverse=True)",
         "confidence": 0.9,
         "code_hint": "return sorted(data, reverse=True)",
         "approach": "builtin"},
        {"id": "bubble_sort", "desc": "Bubble sort descending implementation",
         "confidence": 0.75,
         "code_hint": "arr = data[:]\nfor i in range(len(arr)):\n    for j in range(len(arr)-1-i):\n        if arr[j] < arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\nreturn arr",
         "approach": "bubble_sort"},
        {"id": "heap", "desc": "Use heapq for descending order",
         "confidence": 0.8,
         "code_hint": "import heapq\nreturn heapq.nlargest(len(data), data)",
         "approach": "heapq"},
    ],
    "sort_ascending": [
        {"id": "builtin", "desc": "Use Python sorted()",
         "confidence": 0.9,
         "code_hint": "return sorted(data)",
         "approach": "builtin"},
        {"id": "bubble_sort", "desc": "Bubble sort ascending implementation",
         "confidence": 0.75,
         "code_hint": "arr = data[:]\nfor i in range(len(arr)):\n    for j in range(len(arr)-1-i):\n        if arr[j] > arr[j+1]: arr[j], arr[j+1] = arr[j+1], arr[j]\nreturn arr",
         "approach": "bubble_sort"},
        {"id": "insertion_sort", "desc": "Insertion sort implementation",
         "confidence": 0.75,
         "code_hint": "arr = data[:]\nfor i in range(1, len(arr)):\n    key = arr[i]\n    j = i - 1\n    while j >= 0 and arr[j] > key:\n        arr[j+1] = arr[j]\n        j -= 1\n    arr[j+1] = key\nreturn arr",
         "approach": "insertion_sort"},
    ],
    "calculate_average": [
        {"id": "builtin", "desc": "Use sum()/len()",
         "confidence": 0.9,
         "code_hint": "return sum(data) / len(data)",
         "approach": "builtin"},
        {"id": "accumulate", "desc": "Manual accumulation with count",
         "confidence": 0.8,
         "code_hint": "total = 0\nfor x in data:\n    total += x\nreturn total / len(data)",
         "approach": "accumulate"},
    ],
    "remove_duplicates": [
        {"id": "set", "desc": "Use set for deduplication (loses order)",
         "confidence": 0.8,
         "code_hint": "return list(set(data))",
         "approach": "set"},
        {"id": "dict_fromkeys", "desc": "Use dict.fromkeys() (preserves order)",
         "confidence": 0.85,
         "code_hint": "return list(dict.fromkeys(data))",
         "approach": "dict_fromkeys"},
        {"id": "loop_seen", "desc": "Manual loop with seen set (preserves order)",
         "confidence": 0.9,
         "code_hint": "seen = set()\nresult = []\nfor x in data:\n    if x not in seen:\n        seen.add(x)\n        result.append(x)\nreturn result",
         "approach": "loop_seen"},
    ],
    "is_prime": [
        {"id": "trial_division", "desc": "Trial division up to sqrt(n)",
         "confidence": 0.9,
         "code_hint": "if n < 2: return False\nfor i in range(2, int(n**0.5)+1):\n    if n % i == 0: return False\nreturn True",
         "approach": "trial_division"},
        {"id": "miller_rabin", "desc": "Miller-Rabin probabilistic test",
         "confidence": 0.8,
         "code_hint": "import random\ndef is_prime(n, k=5):\n    if n < 2: return False\n    if n == 2 or n == 3: return True\n    if n % 2 == 0: return False\n    r, d = 0, n - 1\n    while d % 2 == 0: r += 1; d //= 2\n    for _ in range(k):\n        a = random.randrange(2, n-1)\n        x = pow(a, d, n)\n        if x == 1 or x == n-1: continue\n        for _ in range(r-1):\n            x = pow(x, 2, n)\n            if x == n-1: break\n        else: return False\n    return True",
         "approach": "miller_rabin"},
    ],
    "filter_above": [
        {"id": "list_comp", "desc": "List comprehension with threshold",
         "confidence": 0.9,
         "code_hint": "return [x for x in data if x > threshold]",
         "approach": "list_comp"},
        {"id": "filter_builtin", "desc": "Use built-in filter()",
         "confidence": 0.8,
         "code_hint": "return list(filter(lambda x: x > threshold, data))",
         "approach": "filter_builtin"},
    ],
    "filter_below": [
        {"id": "list_comp", "desc": "List comprehension with threshold",
         "confidence": 0.9,
         "code_hint": "return [x for x in data if x < threshold]",
         "approach": "list_comp"},
        {"id": "filter_builtin", "desc": "Use built-in filter()",
         "confidence": 0.8,
         "code_hint": "return list(filter(lambda x: x < threshold, data))",
         "approach": "filter_builtin"},
    ],
    "generate_range": [
        {"id": "range", "desc": "Use range() with start/end",
         "confidence": 0.9,
         "code_hint": "return list(range(start, end + 1))",
         "approach": "range"},
        {"id": "list_comp", "desc": "List comprehension range",
         "confidence": 0.85,
         "code_hint": "return [i for i in range(start, end + 1)]",
         "approach": "list_comp"},
    ],
}


class HypothesisGenerator:
    """Generates multiple solution hypotheses from semantic understanding.

    Each hypothesis represents a distinct approach:
    - Different algorithmic strategies (not just structural variations)
    - Different code structures
    - Different levels of complexity
    - Different implementation patterns

    Non-determinism comes from the system choosing different valid strategies
    based on context, not from random string selection.
    """

    def __init__(self, config: Optional[HypothesisConfig] = None):
        self.config = config or HypothesisConfig()

    def generate(
        self,
        operations: list[str],
        entities: list[dict],
        goal_type: str,
        composition: dict,
        context: Optional[dict] = None,
    ) -> tuple[list[Hypothesis], list[GenerationEvent]]:
        """Generate multiple hypotheses from semantic analysis.

        Each hypothesis uses a different algorithmic strategy for the same
        operations. The strategies are selected from the knowledge of valid
        approaches, not hardcoded answers.

        Returns (hypotheses, events) where events document the process.
        """
        events = []
        hypotheses = []

        events.append(GenerationEvent(
            event_type=EventType.HYPOTHESIS_STARTED,
            metadata={"operations": operations, "goal": goal_type},
        ))

        # For each operation, find available strategies
        op_strategies = {}
        for op in operations:
            strategies = _ALGORITHMIC_STRATEGIES.get(op, [])
            if strategies:
                op_strategies[op] = strategies

        if not op_strategies:
            # No known strategies for these operations - use composition directly
            h = self._hypothesis_direct_composition(operations, entities, goal_type, composition)
            if h.confidence >= self.config.min_confidence:
                hypotheses.append(h)
        else:
            # Generate hypotheses using different strategy combinations
            hypotheses = self._generate_strategy_hypotheses(
                operations, op_strategies, entities, goal_type, composition
            )

        for h in hypotheses:
            events.append(GenerationEvent(
                event_type=EventType.HYPOTHESIS_GENERATED,
                hypothesis_id=h.id,
                content=h.description,
                metadata={"approach": h.approach, "confidence": h.confidence,
                          "operations": h.operations},
            ))

        events.append(GenerationEvent(
            event_type=EventType.SELECTING,
            content=f"Generated {len(hypotheses)} hypotheses",
            metadata={"count": len(hypotheses)},
        ))

        # The think budget: a deeper request pays for MORE candidates —
        # a normal request keeps exactly the configured cap.
        from brain.think_budget import current_budget
        budget = current_budget()
        cap = (budget.max_hypotheses if budget.mode.value != "normal"
               else self.config.max_hypotheses)
        return hypotheses[:cap], events

    def _generate_strategy_hypotheses(
        self,
        operations: list[str],
        op_strategies: dict[str, list[dict]],
        entities: list[dict],
        goal_type: str,
        composition: dict,
    ) -> list[Hypothesis]:
        """Generate hypotheses using different strategy combinations.

        Uses fingerprinting to avoid duplicate hypotheses. Each hypothesis
        is uniquely identified by its strategy choices across all operations.
        """
        hypotheses = []
        seen_fingerprints = set()

        # Strategy 1: Use the highest-confidence strategy for each operation
        h1 = self._build_hypothesis(
            id=1,
            strategy_label="optimal",
            operations=operations,
            op_strategies=op_strategies,
            strategy_choice=lambda ops: {op: strats[0]["id"]
                                          for op, strats in ops.items()},
            confidence_boost=0.0,
        )
        if h1.confidence >= self.config.min_confidence:
            fp = self._fingerprint(h1)
            if fp not in seen_fingerprints:
                seen_fingerprints.add(fp)
                hypotheses.append(h1)

        # Strategy 2: Use an alternative strategy for the most complex operation
        if len(operations) >= 1 and self.config.explore_alternatives:
            # Find the operation with the most alternatives
            most_complex = max(op_strategies.items(),
                             key=lambda x: len(x[1]), default=None)
            if most_complex and len(most_complex[1]) > 1:
                alt_strat = most_complex[1][1]  # Second strategy
                h2 = self._build_hypothesis(
                    id=2,
                    strategy_label=alt_strat["approach"],
                    operations=operations,
                    op_strategies=op_strategies,
                    strategy_choice=lambda ops, alt=alt_strat, mc=most_complex: {
                        op: (alt["id"] if op == mc[0] else strats[0]["id"])
                        for op, strats in ops.items()
                    },
                    confidence_boost=-0.05,
                )
                if h2.confidence >= self.config.min_confidence:
                    fp = self._fingerprint(h2)
                    if fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        hypotheses.append(h2)

        # Strategy 3: If there are 3+ operations, try a different combination
        if len(operations) >= 3 and self.config.explore_alternatives:
            third_strat = None
            for op, strats in op_strategies.items():
                if len(strats) >= 2:
                    third_strat = (op, strats[1])
                    break
            if third_strat:
                h3 = self._build_hypothesis(
                    id=3,
                    strategy_label=third_strat[1]["approach"],
                    operations=operations,
                    op_strategies=op_strategies,
                    strategy_choice=lambda ops, ts=third_strat: {
                        op: (ts[1]["id"] if op == ts[0] else strats[0]["id"])
                        for op, strats in ops.items()
                    },
                    confidence_boost=-0.1,
                )
                if h3.confidence >= self.config.min_confidence:
                    fp = self._fingerprint(h3)
                    if fp not in seen_fingerprints:
                        seen_fingerprints.add(fp)
                        hypotheses.append(h3)

        return hypotheses

    def _fingerprint(self, hypothesis: Hypothesis) -> str:
        """Create a fingerprint for a hypothesis based on its strategy choices.

        Two hypotheses with the same fingerprint are effectively duplicates
        and only one should be kept.
        """
        strategies = hypothesis.structure.get("strategies", {})
        # Sort by operation name for consistent fingerprinting
        parts = [f"{op}={strat}" for op, strat in sorted(strategies.items())]
        return "|".join(parts) if parts else hypothesis.approach

    def _build_hypothesis(
        self,
        id: int,
        strategy_label: str,
        operations: list[str],
        op_strategies: dict[str, list[dict]],
        strategy_choice,
        confidence_boost: float = 0.0,
    ) -> Hypothesis:
        """Build a hypothesis from chosen strategies."""
        choices = strategy_choice(op_strategies)

        # Build description from chosen strategies
        strat_descs = []
        total_confidence = 0.0
        chosen_strategies = []
        for op in operations:
            strat_id = choices.get(op)
            strats = op_strategies.get(op, [])
            chosen = next((s for s in strats if s["id"] == strat_id), strats[0] if strats else None)
            if chosen:
                strat_descs.append(f"{op}: {chosen['desc']}")
                total_confidence += chosen["confidence"]
                chosen_strategies.append(chosen)
            else:
                strat_descs.append(f"{op}: default implementation")
                total_confidence += 0.5

        avg_confidence = (total_confidence / max(len(operations), 1)) + confidence_boost

        # Build code hints
        code_hints = [s.get("code_hint", "") for s in chosen_strategies if s.get("code_hint")]

        return Hypothesis(
            id=id,
            description=f"Strategy [{strategy_label}]: {' | '.join(strat_descs)}",
            approach=strategy_label,
            operations=operations,
            structure={
                "type": "strategy_based",
                "strategies": {op: choices.get(op, "default") for op in operations},
                "code_hints": code_hints,
            },
            confidence=min(max(avg_confidence, 0.0), 1.0),
            reasoning_path=["algorithmic_strategies", strategy_label],
        )

    def _hypothesis_direct_composition(
        self, operations: list[str], entities: list[dict],
        goal_type: str, composition: dict
    ) -> Hypothesis:
        """Fallback: Follow the composition graph directly."""
        graph = composition.get("graph", {})
        ops = graph.get("operations", [])
        op_names = [op.get("name", op) if isinstance(op, dict) else op for op in ops] if ops else operations

        return Hypothesis(
            id=1,
            description=f"Direct implementation: {' → '.join(op_names)}",
            approach="composition_direct",
            operations=op_names,
            structure={"type": "sequential", "ops": op_names},
            confidence=0.8 if op_names else 0.4,
            reasoning_path=["composition_graph", "direct_execution_order"],
        )

    def select(
        self, hypotheses: list[Hypothesis], context: Optional[dict] = None
    ) -> tuple[Optional[Hypothesis], list[GenerationEvent]]:
        """Select the best hypothesis. Returns (selected, events)."""
        events = []
        if not hypotheses:
            return None, events

        # Score each hypothesis
        scored = []
        for h in hypotheses:
            score = self._score_hypothesis(h, context)
            scored.append((score, h))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        events.append(GenerationEvent(
            event_type=EventType.HYPOTHESIS_SELECTED,
            hypothesis_id=best.id,
            content=best.description,
            metadata={"selected_id": best.id, "total": len(hypotheses),
                      "score": scored[0][0]},
        ))

        return best, events

    def _score_hypothesis(self, h: Hypothesis, context: Optional[dict]) -> float:
        """Score a hypothesis for selection."""
        score = h.confidence
        # Prefer higher-confidence strategies
        if h.approach == "builtin":
            score += 0.1
        # Prefer implementations that preserve order when relevant
        if context and context.get("preserve_order"):
            if "loop_seen" in h.approach or "dict_fromkeys" in h.approach:
                score += 0.05
        return score
